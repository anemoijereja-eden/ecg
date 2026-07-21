E.D.E.N.Tech Open Source Hardware health telemetry system
This project aims to bring the features of medical grade wireless telemetry systems to the people, making them vastly more compact and lowering costs significantly.
Where the average medical grade device that does what this unit can costs hundreds of dollars and only supports proprietary EMR systems, this device uses commodity parts and is intended to connect to an open source telemetry system based on InfluxDB.

Current work completed:
- Signal acquisition using an esp32-s3
- Off device signal processing
- first revision PCB design

To Do:
- Move signal processing onto the device itself
- enable device to report data over the network
- finish PCB designs and create the actual device
- create software solution for processing telemetry data on a recieving server
- Look into other telemetry to integrate

Potential ideas:
- non-invasive blood pressure monitor built more compact than ever before using a PCB motor, integrated cycloidal drive, and a membrane pump to create the full system at a pcb scale.
- integration with implanted sensors via nfc

Tech Stack:
The system is based around the AD8232 as an analog frontend, connected to an ESP32-S3 microcontroller to create a low cost platform using parts that are already widely used and accepted in open hardware.
This repository contains an esp32 firmware, a kicad project under the "PCB" subdirectory, and the prototype off-device processing scripts in "plotter.py"
