from comfy_api.latest import io


class SizeConstraint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SizeConstraint", display_name="Size Constraint", category="TensorRT/detailers",
            description="Requires Impact Pack. Resizes each detailer crop to exactly width × height before encoding, without duplicating the batch. Match the TensorRT engine resolution.",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=8),
                io.Int.Input("height", default=1024, min=64, max=16384, step=8),
            ],
            outputs=[io.Custom("DETAILER_HOOK").Output(display_name="DETAILER_HOOK")],
        )

    @classmethod
    def execute(cls, width, height):
        # Impact Pack may register after TensorRTNodes during custom-node loading.
        from impact.hooks import DetailerHook

        if width <= 0 or height <= 0 or width % 8 or height % 8:
            raise ValueError("Size Constraint width and height must be positive multiples of 8.")

        class SizeConstraintHook(DetailerHook):
            def __init__(self, width, height):
                super().__init__()
                self.width = width
                self.height = height

            def touch_scaled_size(self, w, h):
                return self.width, self.height

        return io.NodeOutput(SizeConstraintHook(width, height))


NODE_CLASS_MAPPINGS = {"SizeConstraint": SizeConstraint}
NODE_DISPLAY_NAME_MAPPINGS = {"SizeConstraint": "Size Constraint"}
