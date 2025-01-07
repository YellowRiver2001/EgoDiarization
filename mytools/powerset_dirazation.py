def get_embeddings(
    self,
    file,  # 音频文件
    binary_segmentations: SlidingWindowFeature,  # 二值化后的分段信息，形状为 (num_chunks, num_frames, num_speakers)
    exclude_overlap: bool = False,  # 是否排除重叠的语音区域
    hook: Optional[Callable] = None,  # 在每个批次处理后调用的回调函数
):
    """提取每个 (chunk, speaker) 对的嵌入特征

    参数
    ----------
    file : AudioFile
        音频文件。
    binary_segmentations : SlidingWindowFeature
        二值化后的分段信息，形状为 (num_chunks, num_frames, num_speakers)。
    exclude_overlap : bool, optional
        是否排除重叠的语音区域。当非重叠语音太短时，会使用整个语音段。
    hook: Optional[Callable]
        每批处理后调用的回调函数，用于报告进度。

    返回
    -------
    embeddings : np.ndarray
        每个 (chunk, speaker) 对应的嵌入特征，形状为 (num_chunks, num_speakers, dimension)。
    """

    # 如果是训练模式，检查是否可以使用缓存的嵌入以加速优化过程
    if self.training:
        # 获取缓存的嵌入特征
        cache = file.get("training_cache/embeddings", dict())
        # 如果缓存存在且分割模型使用 powerset 模式，或者阈值相同，则复用缓存的嵌入
        if ("embeddings" in cache) and (
            self._segmentation.model.specifications.powerset
            or (cache["segmentation.threshold"] == self.segmentation.threshold)
        ):
            return cache["embeddings"]

    # 获取每段的持续时间
    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

    # 如果需要排除重叠区域
    if exclude_overlap:
        # 最小的采样数，以避免嵌入提取错误
        min_num_samples = self._embedding.min_num_samples
        # 对应的最小帧数
        num_samples = duration * self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

        # 标记不重叠的帧，zero-out 重叠的帧
        clean_frames = 1.0 * (
            np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
        )
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data * clean_frames,
            binary_segmentations.sliding_window,
        )
    else:
        min_num_frames = -1
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data, binary_segmentations.sliding_window
        )

    # 生成音频段和相应的掩码
    def iter_waveform_and_mask():
        for (chunk, masks), (_, clean_masks) in zip(
            binary_segmentations, clean_segmentations
        ):
            # 提取每段音频的波形
            waveform, _ = self._audio.crop(
                file,
                chunk,
                duration=duration,
                mode="pad",
            )

            # 将掩码中的 NaN 替换为 0
            masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

            # 根据是否满足最小帧数来选择掩码
            for mask, clean_mask in zip(masks.T, clean_masks.T):
                if np.sum(clean_mask) > min_num_frames:
                    used_mask = clean_mask
                else:
                    used_mask = mask
                yield waveform[None], torch.from_numpy(used_mask)[None]

    # 分批处理数据
    batches = batchify(
        iter_waveform_and_mask(),
        batch_size=self.embedding_batch_size,
        fillvalue=(None, None),
    )

    # 计算批次数
    batch_count = math.ceil(num_chunks * num_speakers / self.embedding_batch_size)
    embedding_batches = []

    # 调用 hook 以报告进度
    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)

    # 提取每批次的嵌入特征
    for i, batch in enumerate(batches, 1):
        waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

        waveform_batch = torch.vstack(waveforms)
        mask_batch = torch.vstack(masks)

        # 获取嵌入特征
        embedding_batch: np.ndarray = self._embedding(
            waveform_batch, masks=mask_batch
        )
        embedding_batches.append(embedding_batch)

        # 更新 hook 的进度
        if hook is not None:
            hook("embeddings", embedding_batch, total=batch_count, completed=i)

    # 将所有批次的嵌入特征合并
    embedding_batches = np.vstack(embedding_batches)

    # 重新排列嵌入特征的维度
    embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

    # 如果是训练模式，缓存嵌入特征
    if self.training:
        if self._segmentation.model.specifications.powerset:
            file["training_cache/embeddings"] = {
                "embeddings": embeddings,
            }
        else:
            file["training_cache/embeddings"] = {
                "segmentation.threshold": self.segmentation.threshold,
                "embeddings": embeddings,
            }

    return embeddings  # 返回嵌入特征


def reconstruct(
    self,
    segmentations: SlidingWindowFeature,  # 说话者分割信息，形状为 (num_chunks, num_frames, num_speakers)
    hard_clusters: np.ndarray,  # 聚类结果，形状为 (num_chunks, num_speakers)
    count: SlidingWindowFeature,  # 活跃说话者数量，形状为 (total_num_frames, 1)
) -> SlidingWindowFeature:
    """根据聚类后的分割信息构建最终的离散化说话者分离结果

    参数
    ----------
    segmentations : SlidingWindowFeature
        原始说话者分割信息，形状为 (num_chunks, num_frames, num_speakers)。
    hard_clusters : np.ndarray
        聚类结果，形状为 (num_chunks, num_speakers)。
    count : SlidingWindowFeature
        活跃说话者数量，形状为 (total_num_frames, 1)。

    返回
    -------
    discrete_diarization : SlidingWindowFeature
        最终的离散化（0 和 1）的说话者分离结果。
    """

    num_chunks, num_frames, local_num_speakers = segmentations.data.shape
    num_clusters = np.max(hard_clusters) + 1  # 获取聚类的数量

    # 初始化聚类后的分割结果，初始值为 NaN
    clustered_segmentations = np.NAN * np.zeros(
        (num_chunks, num_frames, num_clusters)
    )

    # 遍历每个分块和对应的聚类结果
    for c, (cluster, (chunk, segmentation)) in enumerate(
        zip(hard_clusters, segmentations)
    ):
        # cluster 的形状为 (local_num_speakers,)
        # segmentation 的形状为 (num_frames, local_num_speakers)
        for k in np.unique(cluster):
            if k == -2:  # 跳过未分配的聚类
                continue

            # 将属于同一聚类的说话者的分割结果取最大值
            clustered_segmentations[c, :, k] = np.max(
                segmentation[:, cluster == k], axis=1
            )

    # 将聚类后的分割信息转换为 SlidingWindowFeature 格式
    clustered_segmentations = SlidingWindowFeature(
        clustered_segmentations, segmentations.sliding_window
    )

    # 调用 to_diarization 方法将聚类结果转换为最终的离散化说话者分离结果
    return self.to_diarization(clustered_segmentations, count)


def apply(
    self,
    file: AudioFile,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_embeddings: bool = False,
    hook: Optional[Callable] = None,
) -> Annotation:
    """
    应用说话人分离

    参数
    ----------
    file : AudioFile
        已处理的音频文件。
    num_speakers : int, optional
        已知的说话人数。
    min_speakers : int, optional
        最小说话人数。当 `num_speakers` 已知时无效。
    max_speakers : int, optional
        最大说话人数。当 `num_speakers` 已知时无效。
    return_embeddings : bool, optional
        是否返回代表性说话人嵌入。
    hook : callable, optional
        在管道的每个主要步骤之后调用的回调函数，提供步骤名称、生成的工件和处理的文件。
        
    返回
    -------
    diarization : Annotation
        说话人分离结果。
    embeddings : np.array, optional
        当 `return_embeddings` 为 True 时，返回每个说话人的嵌入向量。
    """

    # 设置 hook（例如用于调试目的）
    hook = self.setup_hook(file, hook=hook)

    # 设置说话人数
    num_speakers, min_speakers, max_speakers = self.set_num_speakers(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # 获取分割结果
    segmentations = self.get_segmentations(file, hook=hook)
    hook("segmentation", segmentations)
    #   形状: (num_chunks, num_frames, local_num_speakers)

    # 二值化分割结果
    if self._segmentation.model.specifications.powerset:
        binarized_segmentations = segmentations
    else:
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations,
            onset=self.segmentation.threshold,
            initial_state=False,
        )

    # 估算每帧说话人数
    count = self.speaker_count(
        binarized_segmentations,
        self._segmentation.model.receptive_field,
        warm_up=(0.0, 0.0),
    )
    hook("speaker_counting", count)
    #   形状: (num_frames, 1)
    #   数据类型: int

    # 若没有说话人活动，提前退出
    if np.nanmax(count.data) == 0.0:
        diarization = Annotation(uri=file["uri"])
        if return_embeddings:
            return diarization, np.zeros((0, self._embedding.dimension))

        return diarization

    # 如果使用 OracleClustering 且不返回嵌入
    if self.klustering == "OracleClustering" and not return_embeddings:
        embeddings = None
    else:
        # 获取嵌入
        embeddings = self.get_embeddings(
            file,
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
            hook=hook,
        )
        hook("embeddings", embeddings)
        #   形状: (num_chunks, local_num_speakers, dimension)

    # 聚类获取说话人硬标签和质心
    hard_clusters, _, centroids = self.clustering(
        embeddings=embeddings,
        segmentations=binarized_segmentations,
        num_clusters=num_speakers,
        min_clusters=min_speakers,
        max_clusters=max_speakers,
        file=file,  # 用于 Oracle 聚类
        frames=self._segmentation.model.receptive_field,  # 用于 Oracle 聚类
    )
    # hard_clusters: (num_chunks, num_speakers)
    # centroids: (num_speakers, dimension)   

    # 检测到的不同说话人数
    num_different_speakers = np.max(hard_clusters) + 1

    # 检查说话人数是否超出设定范围
    if (
        num_different_speakers < min_speakers
        or num_different_speakers > max_speakers
    ):
        warnings.warn(
            textwrap.dedent(
                f"""
            检测到的说话人数 ({num_different_speakers}) 超出设定范围 [{min_speakers}, {max_speakers}]。
            如果音频文件太短，可能无法容纳 {min_speakers} 或更多说话人，请尝试降低最小说话人数。
            """
            )
        )

    # 限制最大瞬时说话人数为 `max_speakers`
    count.data = np.minimum(count.data, max_speakers).astype(np.int8)

    # 从硬聚类结果重构离散说话人分离结果

    # 跟踪不活跃的说话人
    inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
    #   形状: (num_chunks, num_speakers)

    hard_clusters[inactive_speakers] = -2
    discrete_diarization = self.reconstruct(
        segmentations,
        hard_clusters,
        count,
    )
    hook("discrete_diarization", discrete_diarization)

    # 转换为连续的说话人分离结果
    diarization = self.to_annotation(
        discrete_diarization,
        min_duration_on=0.0,
        min_duration_off=self.segmentation.min_duration_off,
    )
    diarization.uri = file["uri"]

    # 此时，说话人标签为 0 到 `num_speakers - 1` 的整数，且与 `centroids` 的行对齐。

    # 如果提供了参考标注，使用其映射假设的说话人到参考说话人
    if "annotation" in file and file["annotation"]:
        _, mapping = self.optimal_mapping(
            file["annotation"], diarization, return_mapping=True
        )

        # 如果假设中的说话人多于参考中的说话人，将这些额外说话人补充到映射中
        mapping = {key: mapping.get(key, key) for key in diarization.labels()}

    else:
        # 当没有参考时，将假设中的说话人重命名为可读格式 SPEAKER_00, SPEAKER_01, ...
        mapping = {
            label: expected_label
            for label, expected_label in zip(diarization.labels(), self.classes())
        }

    # 重命名标签
    diarization = diarization.rename_labels(mapping=mapping)

    # 此时，说话人标签是字符串格式（或当存在参考时，部分标签为字符串和整数的混合）

    if not return_embeddings:
        return diarization

    # 如果使用 OracleClustering，质心可能为 None
    if centroids is None:
        return diarization, None

    # 如果嵌入的质心数量小于说话人数量，补零嵌入
    if len(diarization.labels()) > centroids.shape[0]:
        centroids = np.pad(
            centroids, ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0))
        )

    # 根据说话人标签重新排序质心
    inverse_mapping = {label: index for index, label in mapping.items()}
    centroids = centroids[
        [inverse_mapping[label] for label in diarization.labels()]
    ]

    return diarization, centroids
