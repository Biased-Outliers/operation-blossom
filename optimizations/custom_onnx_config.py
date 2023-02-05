

class ViTOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "img"}),
                # ("attention_mask", {0: "batch", 1: "sequence"}),
                # ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"}), ("pooler_output", {0: "batch"})]),