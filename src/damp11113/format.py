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

