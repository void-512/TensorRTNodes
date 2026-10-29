import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/intermediate_storage.js", import.meta.url), "utf8")
    .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.app;");

function createNode(available) {
    let extension;
    vm.runInNewContext(source, { app: { registerExtension: (value) => { extension = value; } } });
    let callbacks = 0;
    let configurations = 0;
    class Node {
        constructor() {
            this.widgets = [
                { name: "keep_intermediates_in_ram", value: false, options: {}, callback() { callbacks++; return "callback"; } },
                { name: "intermediate_directory", value: "selected/path", options: {} },
            ];
        }
        onNodeCreated() { return "created"; }
        onConfigure() { configurations++; return "configured"; }
        setDirtyCanvas() {}
    }
    extension.beforeRegisterNodeDef(Node, {
        name: "ExportTensorRTPackage",
        input: { optional: { keep_intermediates_in_ram: ["BOOLEAN", { read_only: !available }] } },
    });
    const node = new Node();
    assert.equal(node.onNodeCreated(), "created");
    return { node, ram: node.widgets[0], directory: node.widgets[1], counts: () => ({ callbacks, configurations }) };
}

function assertLocked(widget, expected) {
    assert.equal(widget.disabled, expected);
    assert.equal(widget.options.disabled, expected);
    assert.equal(widget.options.read_only, expected);
}

test("missing shared memory locks RAM off and leaves the directory editable", () => {
    const { ram, directory } = createNode(false);
    assert.equal(ram.value, false);
    assertLocked(ram, true);
    assertLocked(directory, false);
});

test("enabling RAM locks the directory; disabling RAM restores its value", () => {
    const { ram, directory, counts } = createNode(true);
    assertLocked(ram, false);
    assertLocked(directory, false);
    ram.value = true;
    assert.equal(ram.callback(), "callback");
    assertLocked(directory, true);
    ram.value = false;
    ram.callback();
    assertLocked(directory, false);
    assert.equal(directory.value, "selected/path");
    assert.equal(counts().callbacks, 2);
});

test("loading a RAM workflow restores the locked directory", () => {
    const { node, ram, directory, counts } = createNode(true);
    ram.value = true;
    assert.equal(node.onConfigure(), "configured");
    assertLocked(directory, true);
    assert.equal(counts().configurations, 1);
});

test("loading a RAM workflow on an unsupported host turns RAM off", () => {
    const { node, ram, directory } = createNode(false);
    ram.value = true;
    node.onConfigure();
    assert.equal(ram.value, false);
    assertLocked(ram, true);
    assertLocked(directory, false);
});
