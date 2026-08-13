# Folia Embedded

Folia Embedded contains the firmware and tooling used to run the Folia QP prosthetic
ankle along with the Python-based logging and control utilities used during
development, testing, and lab deployments.

This repository is organized to keep embedded firmware, drivers, generated
protocol buffers, and host-side tooling together so firmware and analysis stay
in sync.

Quick overview

- embedded/: Firmware, drivers, and PlatformIO project used to build and flash
	the device.
- pyviz/: Python tools for logging, commanding, and visualizing data from the
	device (serial/USB interface). Contains a small Python package and scripts.
- proto/: Protocol buffer definitions used by embedded and host tooling.
- gen/: Generated protobuf bindings for Python and C used by firmware and
	host tools.

Requirements

- PlatformIO (to build/flash the firmware)
- Python 3.8+ (3.11 recommended) and pip for pyviz tools
- A serial/USB adapter or the device's native USB port for communication
- Optional: Edge Impulse SDK and toolchain for models placed in
	embedded/lib/ei_model when re-training or re-exporting models

Building and flashing firmware (embedded)

1. Install PlatformIO and the appropriate toolchain for your board.
2. Open the `embedded/` folder in your IDE or use the PlatformIO CLI.

Example (CLI):

```bash
cd embedded
platformio run           # build firmware
platformio run -t upload # flash to device (choose env in platformio.ini)
```

If you use an IDE (VS Code + PlatformIO), open the `embedded/` folder and use
the integrated build/upload buttons. If you use Zed and clangd, remember to
update `.zed/settings.json` with your `pio` toolchain location.

Firmware layout notes

- `embedded/src/` contains the main application sources.
- `embedded/lib/` contains local libraries and drivers (ADS1220 ADC driver,
	gait filter, Edge Impulse runtime and model artifacts).
- `embedded/platformio.ini` contains board/environment definitions and build
	flags used to produce firmware binaries for specific hardware revisions.

Host tools and logging (pyviz)

The `pyviz/` folder contains Python scripts and modules used to:

- connect to the running device over serial/USB
- stream and store logged sensor and state data
- send control/command messages (for testing and calibration)
- provide simple visualization and quick analysis helpers

Tooling: `uv` (project venv + package manager)

This project uses `uv` as the virtual environment and package manager for the
Python host tools. Before running any logger or host-side script, synchronize
the project environment and dependencies with:

```bash
uv sync
```

Run host tools (loggers and scripts) using `uv run` so they execute inside the
managed environment. Important prerequisites: power the device and connect the
host computer to the FoliaDriver WiFi network before starting any logging
script that communicates over the device network interface.

Examples:

```bash
# Start the primary network logger (must specify device IP)
uv run logger.py --device-ip 192.168.4.1

# Start alternate loggers (no device IP required in the run call)
uv run logger2.py
uv run logger3.py

# Run an interactive analyzer or notebook runner
uv run python pyviz/analyze_notebook.py
```

If you prefer not to use `uv`, you can still create a virtual environment and
install dependencies manually, but ensure the same dependencies are present as
defined by your `uv` configuration.


Protocol buffers and generated code

- `proto/Packet.proto` holds the canonical message definitions. If you edit
	those files you must re-generate the bindings used by firmware and host
	tools. Generated artifacts are kept in `gen/`.
- We use `buf` to manage and generate protobuf artifacts across the project.
	Make sure `buf` is installed and run the generator from the repository root:

```bash
buf generate
```

- To regenerate Python bindings directly with `protoc` (if you prefer):

```bash
protoc --python_out=gen/ proto/Packet.proto
```

Development notes

- Keep `proto/Packet.proto` and the generated sources in `gen/` in sync. When
	changing any message formats, update both the embedded and host sides and
	bump any version identifiers used in the communication protocol.
- Edge Impulse artifacts and a compiled model live under
	`embedded/lib/ei_model`. Replacing or re-exporting models requires following
	Edge Impulse export steps and copying the resulting sources into that
	folder.
- Drivers (for example the ADS1220 ADC) are in `embedded/lib/ads1220_driver`.
	Driver changes can affect ADC timing and calibration — test carefully.

Running tests and experiments

- Unit tests and host-side experiments live in `pyviz/` and the top-level
	`pyviz` scripts. Use the provided notebooks and small scripts to reproduce
	collection workflows.
- Embedded test code and examples are under `embedded/test/`.

Troubleshooting

- Serial port missing: check device power, USB cable, and correct drivers for
	your platform. On Windows use Device Manager to find the COM port.
- Build failures: ensure PlatformIO has the correct board environment and
	toolchain; inspect `embedded/platformio.ini` for environment definitions.
- Model inference differences: confirm the same model binary and runtime are
	used on both the host (for offline tests) and the embedded runtime.

Contributing

If you contribute changes that affect communication messages, include a
backwards-compatible migration plan or bump the protocol version in
`proto/Packet.proto` and update generated code.

License

See the repository `LICENSE` for license terms.

Contact and notes

If you need help reproducing experiments or wiring the device, include logs
from `pyviz/logger.py` and the firmware binary version when opening issues.

