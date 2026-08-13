"""Отдельный процесс: размечает, кто и когда говорит.

Запускается основным приложением как `trnslt.exe --diarize запись N отчёт`.
Своим процессом — потому что sherpa-onnx держит блокировку интерпретатора всё
время расчёта: в одном процессе с окном оно замирает на минуты.

Модуль намеренно не трогает faster-whisper и CUDA: их загрузка в дочернем
процессе стоила бы около минуты впустую на каждый файл.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import (MAX_SPEAKERS, MERGE_THRESHOLD, MIN_RELIABLE_SECONDS, SAMPLE_RATE,
                    diarization_models, lower_priority, worker_threads)

# Sherpa просим нарезать запись мельче, чем нужно: его метки мы всё равно
# выбрасываем и расставляем свои, а мелкая нарезка ничего не портит.
OVER_SPLIT = 12


def decode_audio(path: str) -> "object":
    """Читает любой аудио- или видеофайл в моно 16 кГц через PyAV."""
    import av
    import numpy as np

    chunks = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise RuntimeError("в файле нет звуковой дорожки")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(container.streams.audio[0]):
            for piece in resampler.resample(frame):
                chunks.append(piece.to_ndarray().reshape(-1))
        for piece in resampler.resample(None):  # хвост, застрявший в ресемплере
            chunks.append(piece.to_ndarray().reshape(-1))

    if not chunks:
        raise RuntimeError("не удалось прочитать звук")
    return np.concatenate(chunks).astype(np.float32)


def embed_segments(audio, spans: list[tuple[float, float]], model: str):
    """Считает отпечаток голоса для каждого куска речи, отдельно от нарезки."""
    import numpy as np
    import sherpa_onnx

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=worker_threads()))

    vectors = []
    for start, end in spans:
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE,
                               waveform=audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)])
        stream.input_finished()
        vector = np.array(extractor.compute(stream), dtype=np.float64)
        vectors.append(vector / (np.linalg.norm(vector) + 1e-9))
    return np.vstack(vectors)


def merge_voices(vectors, wanted: int):
    """Объединяет отпечатки в голоса по среднему сходству.

    Своя кластеризация вместо встроенной в sherpa: та на разговоре четверых
    склеивала всех в один голос, а на шумной записи, наоборот, плодила два десятка.
    Слияние идёт от самых похожих пар: при заданном числе участников — до него,
    иначе — пока похожесть не упадёт ниже порога.
    """
    import numpy as np

    count = len(vectors)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    sizes = np.ones(count)
    alive = np.ones(count, dtype=bool)
    members: dict[int, list[int]] = {i: [i] for i in range(count)}
    limit = wanted if wanted > 0 else 1

    while alive.sum() > limit:
        pairs = np.where(alive[:, None] & alive[None, :], similarity, -np.inf)
        first, second = np.unravel_index(np.argmax(pairs), pairs.shape)
        if not wanted and pairs[first, second] < MERGE_THRESHOLD:
            break
        if not np.isfinite(pairs[first, second]):
            break

        # Среднее по числу вошедших кусков: связь «average linkage».
        weight_a, weight_b = sizes[first], sizes[second]
        blended = (weight_a * similarity[first] + weight_b * similarity[second]) / (weight_a + weight_b)
        similarity[first, :] = blended
        similarity[:, first] = blended
        similarity[first, first] = -np.inf
        sizes[first] += weight_b
        alive[second] = False
        members[first].extend(members.pop(second))

    return [sorted(group) for group in members.values()]


def assign_speakers(spans, vectors, wanted: int):
    """Расставляет номера голосов по всем кускам, включая слишком короткие.

    Короткие куски не участвуют в подсчёте голосов, но подписать их надо —
    отдаём каждый ближайшему по звучанию центру.
    """
    import numpy as np

    reliable = [i for i, (start, end) in enumerate(spans)
                if end - start >= MIN_RELIABLE_SECONDS]
    if len(reliable) < 2:
        reliable = list(range(len(spans)))

    core = vectors[reliable]
    wanted = min(wanted, len(core)) if wanted > 0 else 0
    groups = merge_voices(core, wanted)
    if not wanted and len(groups) > MAX_SPEAKERS:
        groups = merge_voices(core, MAX_SPEAKERS)

    centres = np.vstack([core[group].mean(axis=0) for group in groups])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True) + 1e-9
    labels = np.argmax(vectors @ centres.T, axis=1)

    # Нумеруем по очереди появления: первый заговоривший — «Спикер 1».
    order: list[int] = []
    for label in labels:
        if label not in order:
            order.append(int(label))
    renumber = {label: index for index, label in enumerate(order)}
    return [renumber[int(label)] for label in labels]


def run(path: str, speakers: int, report_path: str) -> int:
    """Считает разметку и пишет ход работы в файл.

    Общение через файл, а не через стандартный вывод: у собранного приложения
    нет консоли, и писать ему некуда.
    """
    lower_priority()
    report = Path(report_path)

    def write(line: str) -> None:
        with report.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    try:
        import sherpa_onnx

        paths = diarization_models()
        if paths is None:
            raise RuntimeError("не нашлись модели разделения по говорящим")
        segmentation, embedding = paths

        audio = decode_audio(path)
        threads = worker_threads()
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation)),
                num_threads=threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding), num_threads=threads),
            # Метки sherpa не используем — важна только нарезка на куски речи.
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=OVER_SPLIT),
            min_duration_on=0.3,   # реплики короче считаем шумом
            min_duration_off=0.5,  # паузы короче не разрывают реплику
        )
        if not config.validate():
            raise RuntimeError("не складывается настройка разделения по говорящим")

        def progress(done: int, total: int) -> int:
            # Нарезка — примерно три четверти работы, остальное отпечатки и группировка.
            write(f"p {0.75 * (done / total if total else 0):.4f}")
            return 0

        segments = sherpa_onnx.OfflineSpeakerDiarization(config).process(
            audio, callback=progress).sort_by_start_time()
        spans = [(s.start, s.end) for s in segments]
        if not spans:
            write(json.dumps({"segments": []}))
            return 0

        write("p 0.80")
        vectors = embed_segments(audio, spans, str(embedding))
        write("p 0.95")
        labels = assign_speakers(spans, vectors, speakers)

        write(json.dumps({"segments": [[start, end, int(label)]
                                       for (start, end), label in zip(spans, labels)]}))
        return 0
    except Exception as exc:
        write(json.dumps({"error": f"{exc.__class__.__name__}: {exc}"}))
        return 1


def main(argv: list[str]) -> int:
    rest = argv[argv.index("--diarize") + 1:]
    return run(rest[0], int(rest[1]), rest[2])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
