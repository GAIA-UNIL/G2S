/* G2S bridge page bootstrap - SPDX-License-Identifier: GPL-3.0-or-later */

import { startG2SLocalBridge } from "./g2s-bridge.js";

const status = document.querySelector("#status");
startG2SLocalBridge({
  onStatus: (_state, message) => { status.textContent = message; },
});
