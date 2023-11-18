"""
damp11113-library - A Utils library and Easy to use. For more info visit https://github.com/damp11113/damp11113-library/wiki
Copyright (C) 2021-2023 damp11113 (MIT)

Visit https://github.com/damp11113/damp11113-library

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import struct

# multiplex4 (m4) format

def create_multiplex4_file(filename, sample_rate, data_format, data_streams):
    with open(filename, 'wb') as file:
        # Write header information
        header = struct.pack('!If', sample_rate, data_format)
        file.write(header)

        # Write data streams
        for stream_data in data_streams:
            metadata = struct.pack('!I', stream_data['id'])  # Example: Stream ID
            file.write(metadata)

            # Write IQ data for each stream
            for iq_sample in stream_data['iq_data']:
                iq_byte = struct.pack('!B', iq_sample)  # Pack the 4-bit IQ sample into a byte
                file.write(iq_byte)


def read_multiplex4_file(file_path):
    with open(file_path, 'rb') as file:
        # Read header information
        header = file.read(8)  # Assuming header is 8 bytes long (4 bytes for sample rate, 4 bytes for format)
        sample_rate, data_format = struct.unpack('!If', header)

        data_streams = []

        # Read data streams
        while True:
            metadata = file.read(4)  # Assuming metadata is 4 bytes long (e.g., stream ID)
            if not metadata:
                break  # Reached the end of the file

            stream_id = struct.unpack('!I', metadata)[0]  # Extract the stream ID

            iq_data = []
            while True:
                iq_byte = file.read(1)  # Assuming each IQ sample is represented by 1 byte (8 bits)
                if not iq_byte:
                    break  # Reached the end of the current data stream

                iq_sample = struct.unpack('!B', iq_byte)[0]  # Unpack the byte as a single 4-bit IQ sample
                iq_data.append(iq_sample)

            data_streams.append({'id': stream_id, 'iq_data': iq_data})

    for stream_data in data_streams:
        iq = '|'.join([str(num) for num in stream_data['iq_data']])
    iqlist = iq.split("|0|0|0")
    iqdi = []
    for id, iqidremove in enumerate(iqlist):
        if id == 0:
            iqdi.append(iqidremove)
        else:
            iqdi.append(iqidremove[3:])
    iqdi2 = []
    for iqreplace in iqdi:
        iqdi2.append(iqreplace.replace('|', ','))
    iqpr = [list(map(int, item.split(','))) for item in iqdi2]
    data_streams = []
    for id, iq in enumerate(iqpr):
        data_streams.append({
            'id': id,
            'iq_data': iq
        })

    return sample_rate, data_format, data_streams

#--------------------------------------------------------------------------------------------------------------

class BrainfuckInterpreter:
    def __init__(self):
        self.memory = [0] * 30000
        self.pointer = 0
        self.output = ""

    def interpret(self, code):
        loop_stack = []
        code_pointer = 0

        while code_pointer < len(code):
            command = code[code_pointer]

            if command == '>':
                self.pointer += 1
            elif command == '<':
                self.pointer -= 1
            elif command == '+':
                self.memory[self.pointer] = (self.memory[self.pointer] + 1) % 256
            elif command == '-':
                self.memory[self.pointer] = (self.memory[self.pointer] - 1) % 256
            elif command == '.':
                self.output += chr(self.memory[self.pointer])
            elif command == ',':
                # Input operation is not implemented in this basic interpreter
                pass
            elif command == '[':
                if self.memory[self.pointer] == 0:
                    loop_depth = 1
                    while loop_depth > 0:
                        code_pointer += 1
                        if code[code_pointer] == '[':
                            loop_depth += 1
                        elif code[code_pointer] == ']':
                            loop_depth -= 1
                else:
                    loop_stack.append(code_pointer)
            elif command == ']':
                if self.memory[self.pointer] != 0:
                    code_pointer = loop_stack[-1] - 1
                else:
                    loop_stack.pop()
            code_pointer += 1

        return self.output