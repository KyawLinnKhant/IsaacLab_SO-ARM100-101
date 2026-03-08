# Copyright (c) 2026, Kyaw Linn Khant
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.ext


# Publicly accessible helper — callable from other extensions via `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    print("[isaac_so_arm101] some_public_function was called with x: ", x)
    return x**x


# Any class extending `omni.ext.IExt` at the top level (listed in `python.modules` of `extension.toml`) is
# instantiated automatically when the extension loads, triggering `on_startup(ext_id)`.
# `on_shutdown()` is called when the extension is disabled or unloaded.
class OmniverseUIExtension(omni.ext.IExt):
    # ext_id identifies this extension instance and can be passed to the extension manager
    # to retrieve metadata such as the filesystem path of this extension.
    def on_startup(self, ext_id):
        print("[isaac_so_arm101] startup")

        self._count = 0

        self._window = omni.ui.Window("My Window", width=300, height=300)
        with self._window.frame:
            with omni.ui.VStack():
                counter_label = omni.ui.Label("")

                def on_click():
                    self._count += 1
                    counter_label.text = f"count: {self._count}"

                def on_reset():
                    self._count = 0
                    counter_label.text = "empty"

                on_reset()

                with omni.ui.HStack():
                    omni.ui.Button("Add", clicked_fn=on_click)
                    omni.ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self):
        print("[isaac_so_arm101] shutdown")