import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "TensorRTNodes.IntermediateStorage",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ExportTensorRTPackage") return;
        const available = !nodeData.input.optional.keep_intermediates_in_ram[1].read_only;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const ram = this.widgets.find((widget) => widget.name === "keep_intermediates_in_ram");
            const directory = this.widgets.find((widget) => widget.name === "intermediate_directory");
            const lock = (widget, disabled) => {
                widget.disabled = disabled;
                widget.options = { ...widget.options, disabled, read_only: disabled };
            };
            const update = () => {
                if (!available && ram.value) ram.value = false;
                lock(ram, !available);
                lock(directory, Boolean(ram.value));
                this.setDirtyCanvas(true, true);
            };
            const callback = ram.callback;
            ram.callback = function () {
                const result = callback?.apply(this, arguments);
                update();
                return result;
            };
            const onConfigure = this.onConfigure;
            this.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                update();
                return result;
            };
            update();
            return result;
        };
    },
});
