#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:01:27 2024

@author: kdavidson
"""
import json
from argparse import ArgumentParser
from collections import OrderedDict
from obspy import UTCDateTime, Stream, Trace
from struct import unpack
import numpy as np

# From Stryde manual
_record_types = {
    2: 'Noise (test record)',
    6: 'IT Shot (direct channel test)',
    8: 'Production Shot (normal record)',
    1: 'Other'
}

_recording_mode = {
    0: 'Normal Mode',
    1: 'Extended Recording Mode'
}

_channel_type_identfication = {
    0x00: 'Unused',
    0x10: 'Seis',
    0x11: 'Electromagnetic (EM)',
    0x20: 'Time break',
    0x21: 'Clock timebreak',
    0x22: 'Field timebreak',
    0x30: 'Up hole',
    0x40: 'Water break',
    0x50: 'Time counter',
    0x60: 'External Data',
    0x61: 'Acoustic range measurement',
    0x62: 'Acoustic reference measured (correlation reference)',
    0x63: 'Acoustic reference nominal (correlation reference)',
    0x70: 'Other',
    0x80: 'Signature/unfiltered',
    0x90: 'Signature/filtered',
    0x91: 'Source signature/unfiltered',
    0x92: 'Source signature/filtered',
    0x93: 'Source signature/estimated',
    0x94: 'Source signature/measured',
    0x95: 'Source base plate',
    0x96: 'Source reference sweep',
    0x97: 'Source other',
    0x98: 'Source reference pilot',
    0x99: 'Source mass',
    0xA0: 'Auxiliary Data Trailer (no longer used)',
    0xB0: 'True reference sweep (correlation reference)',
    0xB1: 'Radio reference sweep',
    0xB2: 'Radio similarity signal',
    0xB3: 'Wireline reference sweep',
    0xC0: 'Depth',
    0xC1: 'Wind',
    0xC2: 'Current',
    0xC3: 'Voltage',
    0xC4: 'Velocity',
    0xC5: 'Acceleration',
    0xC6: 'Pressure',
    0xC7: 'Tension',
    0xC8: 'Tilt measurement',
    0xC9: 'Angle measurement',
    0xF0: 'Calibration trace (time series)',
}

_array_forming = {
    0x01: 'No array forming',
    0x02: '2 groups summed, no weighting.',
    0x03: '3 groups summed, no weighting',
    # etc... see docs
}

_channel_gain_control_method = {
    1: 'Individual AGC',
    2: 'Ganged AGC',
    3: 'Fixed gain',
    4: 'Programmed gain',
    8: 'Binary gain control',
    9: 'IFP gain control'
}

_filter_phase = {
    0: 'Unknown',
    1: 'Minimum',
    2: 'Linear',
    3: 'Zero',
    4: 'Mixed',
    5: 'Maximum'
}

_physical_unit = {
    0x00: 'Unknown',
    0x01: 'Millibar',
    0x02: 'Bar',
    0x03: 'Millimeter/second',
    0x04: 'Meter/second',
    0x05: 'Millimeter/second/second',
    0x06: 'Meter/second/second',
    0x07: 'Newton',
    0x08: 'Kelvin',
    0x09: 'Hertz',
    0x0A: 'Second',
    0x0B: 'Tesla',
    0x0C: 'Volt/meter',
    0x0D: 'Volt meter',
    0x0E: 'Ampere/meter',
    0x0F: 'Volt',
    0x10: 'Ampere',
    0x11: 'Radians (angle)'
    }

_sensor_type = {
    0x00: 'Not defined',
    0x01: 'Hydrophone (pressure sensor)',
    0x02: 'Geophone (velocity sensor) Vertical',
    0x03: 'Geophone, Horizontal, inline',
    0x04: 'Geophone, Horizontal, cross-line',
    0x05: 'Geophone, Horizontal, other',
    0x06: 'Accelerometer, Vertical',
    0x07: 'Accelerometer, Horizontal, inline',
    0x08: 'Accelerometer, Horzontal, crossline',
    0x09: 'Accelerometer, Horizontal, other',
    0x15: 'Electric Dipole',
    0x16: 'Magnetic coil'
}

def int_from_bcd(input_bytes):
    # Check we are working with a bytes object even if single byte input
    if isinstance(input_bytes, int):
        input_bytes = bytes([input_bytes])
    
    a = 0
    n = len(input_bytes) * 2 - 1 # Two digits per byte
    
    for b in input_bytes:
        # Put tens and ones from each byte in correct place
        tens = (b >> 4) & 0xF
        ones = b & 0xF
        # Combine
        a += tens * 10**n + ones * 10**(n - 1)
        n -= 2
        
    return a

def split_byte(input_byte):
    if not (0 <= input_byte <= 255):
        raise ValueError("Input must be a single byte (0-255)")
    
    # High nibble (first four bits)
    high_nibble = (input_byte >> 4) & 0xF
    # Low nibble (second four bits)
    low_nibble = input_byte & 0xF
    
    return high_nibble, low_nibble

# Read the various blocks and so on as in the documentation

def read_ghb1(f):
    buf = f.read(32)
    ghb1 = OrderedDict()
    
    # File number - not needed?
    fn = int_from_bcd(buf[0:2])     # 1,2 - overwritten later
    ghb1['file_number_gbh1'] = fn

    # Format code - always 8036
    ff = int_from_bcd(buf[2:4])     # 3,4
    ghb1['format_code_ghb1'] = ff
    
    # unused 5,6,7,8,9,10
    
    year = int_from_bcd(buf[10:11]) + 2000  # 11
    ghb1['year_ghb1'] = year
    
    # Next byte contains two different things
    nblocks, jday = split_byte(buf[11])
    ghb1['n_additional_blocks_ghb1'] = nblocks # First four bits
    
    jday *= 100     # second four bits, hundreds column in Julian day
    jday += int_from_bcd(buf[12:13]) # 13

    hour = int_from_bcd(buf[13:14])  # 14
    min = int_from_bcd(buf[14:15])   # 15
    sec = int_from_bcd(buf[15:16])   # 16
    
    # Make an ObsPy timestamp for start of record
    # Seconds may be 60 if leap second encountered!
    ghb1['time_ghb1'] = UTCDateTime(year=year, julday=jday,
                               hour=hour, minute=min, second=sec)
    
    # Manufacturer's code - will always be set to 22 for Stryde
    ghb1['manufacturer_code_ghb1'] = int_from_bcd(buf[16:17])     # 17
    
    # Serial number
    ghb1['manufacture_serial_number_ghb1'] = int_from_bcd(buf[17:19]) # 18,19
    
    # 20,21,22 - all zero, unused field in Stryde
    
    bsi = int.from_bytes(buf[22:23]) # 23
    ghb1['base_scan_interval_ghb1'] = bsi # set to binary 0xFF
    
    # 24 - unused by Stryde
    
    rec_type, _ = split_byte(buf[25]) # 25 - first four bits
    ghb1['record_type_ghb1'] = _record_types[rec_type] 
    
    # last half of of 25, 26, 27 unused - set in General header Block 2

    # 28 - scan types per record - always set to 1
    
    # Channel sets per scan type
    ghb1['n_channel_sets_per_scan_ghb1'] = int_from_bcd(buf[28:29]) # 29
    
    # 30 - skew blocks - always set to 0
        
    # Extended header blocks
    ec = int.from_bytes(buf[30:31])     # 31
    if ec == 0xFF:
        ec = None   # real number in General Header Block 2 if set
    ghb1['extended_header_count_ghb1'] = ec  
    
    # External header blocks
    ehl = int.from_bytes(buf[31:32])
    if ehl == 0xFF:
        ehl = None # real number in GHB 2 if set 
    ghb1['external_header_blocks_ghb1'] = ehl
    return ghb1

def read_ghb2(f):
    buf = f.read(32)
    ghb2 = OrderedDict()
    
    ghb2['expanded_file_number_ghb2'] = int.from_bytes(buf[:3])  # 1,2,3
    
    ghb2['extended_channel_sets_ghb2'] = int.from_bytes(buf[3:5]) # 4,5
    
    ghb2['extended_header_blocks_ghb2'] = int.from_bytes(buf[5:8]) # 6,7,8
    
    ghb2['extended_skew_blocks_gbh2'] = int.from_bytes(buf[8:10])  # 9, 10
    
    rev = ord(buf[10:11])        # 11 
    rev += ord(buf[11:12]) / 10.          #12
    ghb2['segd_major_revision_number_ghb2'] = rev
    
    ghb2['generate_trailer_blocks_number'] = int.from_bytes(buf[12:16]) # 13,14,15,16

    ghb2['extended_record_length_in_microseconds_ghb2'] = int.from_bytes(buf[16:20])    # 17,18,19,20

    ghb2['record_set_number'] = int.from_bytes(buf[20:22])    # 21, 22

    ghb2['extended_number_additional_blocks_ghb2'] = int.from_bytes(buf[22:24])     # 23,24
    
    ghb2['dominant_sampling_interval'] = int.from_bytes(buf[24:27])    # 25,26,27
 
    ghb2['external_header_blocks_ghb2'] = int.from_bytes(buf[27:30])    # 28, 29, 30
    
    # 31 - unused

    ghb2['header_block_type_ghb2'] = int.from_bytes(buf[31:]) # 32
    
    return ghb2

def read_ghb3(f):
    buf = f.read(32)
    ghb3 = OrderedDict()
    
    # Time in microseconds after GPS epoch ref
    # Documentation gives this as 'timestamp' so unpack?
    timezero, = unpack('>Q',buf[:8])    # 1,2,3,4,5,6,7,8
    ghb3['GPSTimeZero'] = timezero - 18000000 # Account for leap seconds

            
    ghb3['record_size'] = int.from_bytes(buf[8:16]) # 9,10,11,12,13,14,15,16
    
    ghb3['data_size'] = int.from_bytes(buf[16:24]) # 17,18,19,20,21,22,23,24
    
    # Header size - useful - can be used to skip to start of seismic trace
    ghb3['header_size'] = int.from_bytes(buf[24:28])     # 26,26,27,28
    
    # Look up value above
    erm = int.from_bytes(buf[28:29])    # 29
    ghb3['extended_recording_mode'] = _recording_mode[erm]
    
    ghb3['relative_time_mode'] = int.from_bytes(buf[29:30]) # 30
    
    # 31 - undefined

    ghb3['header_block_type'] = int.from_bytes(buf[31:32]) # 32
    
    return ghb3

def _read_sch(fp):
    buf = fp.read(96)
    sch = OrderedDict()
    
    # 1 - scan type - always set to 1 - not needed

    # channel set number
    sch['channel_set_number'] = int.from_bytes(buf[1:3]) # 2,3
    
    bb = buf[3] # 4 - convert hex value in table above
    sch['channel_type'] = _channel_type_identfication[bb]
    
    tf, = unpack('>i', buf[4:8])        # 5,6,7,8
    sch['start_time'] = tf              
    
    te, = unpack('>i', buf[8:12])       # 9,10,11,12
    sch['end_time'] = te                
 
    sch['number_of_samples'] = int.from_bytes(buf[12:16]) # 13,14,15,16
    
    ds, = unpack('>f', buf[16:20])      # 17,18,19,20
    sch['descale_multiplier'] = ds      # sample descale multiplication factor
    
    sch['channel_count'] = int.from_bytes(buf[20:23])       # 21,22,23
    
    # Sample interval in MICROSECONDS
    sch['SampleInt'] = int.from_bytes(buf[23:26])     # 23,25,26             
    
    ar = buf[26]     # 27
    sch['array_forming'] = _array_forming[ar]           # Array forming

    sch['number_of_trace_headers'] = int.from_bytes(buf[27:28])    # 28
    
    # Byte 29 has two things
    fl, gc = split_byte(buf[28])     # 29
    sch['extended_header_flag'] = fl
    sch['channel_gain_control'] = _channel_gain_control_method[gc]

    sch['vertical_stack'] = int.from_bytes(buf[29:30])     # 30
    
    sch['streamer_cable'] = int.from_bytes(buf[30:31])     # 31

    sch['header_block_type'] = int.from_bytes(buf[31:32])     # 32
    
    af, = unpack('>f', buf[32:36])      # 33,34,35,36
    sch['alias_filter'] = af
    
    lcf, = unpack('>f', buf[36:40])     # 36,37,38,39
    sch['low_cut_filter'] = lcf
    
    afs, = unpack('>f', buf[40:44])     # 40,41,42,43
    sch['alias_filter_slope'] = afs
    
    lcfs, = unpack('>f', buf[44:48])    # 44,45,46,47
    sch['low_cut_filter_slope'] = lcfs
    
    nf1, = unpack('>f', buf[48:52])     # 48,49,50,51
    sch['notch_filter_1'] = nf1 
    
    nf2, = unpack('>f', buf[52:56])     # 52,53,54,55
    sch['notch_filter_2'] = nf1 
    
    nf3, = unpack('>f', buf[56:60])     # 56,57,58,59
    sch['notch_filter_3'] = nf1 
    
    fp = int.from_bytes(buf[60:61])     # 60
    sch['filter_phase'] = _filter_phase[fp]
    
    pu = buf[61]     # 61
    sch['physical_unit'] = _physical_unit[pu]
    
    # 62 - undefined
 
    sch['header_block_type'] = int.from_bytes(buf[63:64])     # 63 

    sch['filter_delay'] = int.from_bytes(buf[64:68])     # 64,65,66,67
    
    desc = buf[68:95].decode('ascii') # 68 - 95
    sch['description'] = desc

    sch['header_block_type_2'] = int.from_bytes(buf[95:96]) # 98
    
    return sch

def read_json_extended_header(f, size):
    # Very generic function to grab a bunch of JSON
    # Parse this out some more?
    buf = f.read(size)
    buf_str = buf.decode('utf-8', errors='ignore')
    start = buf_str.find('{')
    end = buf_str.rfind('}') + 1
    if start == -1 or end == -1:
        print("Could not find a valid JSON block.")
        return None
    json_str = buf_str[start:end]
    json_extended_header = json.loads(json_str)
    return json_extended_header

def read_trace_header(f):   
    #Common trace header for seis and aux
    buf = f.read(20)    # 20 bytes in buffer
    
    trace_hdr = OrderedDict()
    
    # file number again - gets overwritten by extended file number
    fn = int.from_bytes(buf[0:2])       # 1,2
    trace_hdr['file_number'] = fn
    
    st = int.from_bytes(buf[2:3])       # 3
    trace_hdr['scan_type'] = st
    
    cs = int.from_bytes(buf[3:4])       # 4
    trace_hdr['channel_set_number'] = cs
    
    cn = int.from_bytes(buf[4:6])   # 5,6
    trace_hdr['trace_number'] = cn
    
    tm = int.from_bytes(buf[6:9])   #7,8,9
    trace_hdr['timing'] = tm
    
    ehc = int.from_bytes(buf[9:10])     # 10
    trace_hdr['trace_header_extensions_count'] = ehc
    
    sk = int.from_bytes(buf[10:11])     # 11
    trace_hdr['sample_skew'] = sk
    
    te = int.from_bytes(buf[11:12])     # 12
    trace_hdr['trace_edit'] = te
    
    tb = int.from_bytes(buf[12:15])     # 13,14,15
    trace_hdr['time_break'] = tb
    
    ecs = int.from_bytes(buf[15:17])    # 16,17
    trace_hdr['extended_channel_set_number'] = ecs
    
    efn = int.from_bytes(buf[17:20])    # 18,19,20
    trace_hdr['extended_file_number'] = efn
    
    return trace_hdr

def read_traceheader_block1(f):
    blk01 = OrderedDict()
    buf = f.read(32)
    
    rln = int.from_bytes(buf[0:3])      # 1,2,3
    blk01['line_number'] = rln
    
    rpn = int.from_bytes(buf[3:6])      # 4,5,6
    blk01['receiver_number'] = rpn
    
    rpi = int.from_bytes(buf[6:7])      # 7
    blk01['receiver_point_index'] = rpi
    
    rsi = int.from_bytes(buf[7:8])      # 8
    blk01['re-shoot_index'] = rsi
    
    gi = int.from_bytes(buf[8:9])       # 9
    blk01['group_index'] = gi
    
    di = int.from_bytes(buf[9:10])      # 10
    blk01['depth_index'] = di
    
    erln = (int.from_bytes(buf[10:15]))/65536     # 11,12,13,14,15
    blk01['extended_receiver_line_number'] = erln
    
    erpn = (int.from_bytes(buf[15:20]))/65536     # 16,17,18,19,20
    blk01['extended_receiver_point_number'] = erpn
    
    sen = buf[20]    # 21
    blk01['sensor_type'] = _sensor_type[sen]
    
    # don't need the rest of this block now, come back later to complete
    
    return blk01

def read_traceheader_block2(f):
    # Planned sensor position
    blk02 = OrderedDict()
    buf = f.read(32)
    
    x = int.from_bytes(buf[0:4], signed=True)    # 1,2,3,4
    blk02['XLoc'] = x
    
    y = int.from_bytes(buf[4:8], signed=True)   # 5,6,7,8
    blk02['YLoc'] = y
    
    z = int.from_bytes(buf[8:12], signed=True) # 9,10,11,12
    blk02['ZLoc'] = z
    
    ex = int.from_bytes(buf[12:16], signed=True)    # 13,14,15,16
    blk02['LocalX'] = ex
    
    ey = int.from_bytes(buf[16:20], signed=True)    # 17,18,19.20
    blk02['LocalY'] = ey
    
    ez = int.from_bytes(buf[20:24], signed=True)    # 21,22,23,24
    blk02['LocalZ'] = ez
    
    pqa = int.from_bytes(buf[24:28], signed=True)   # 25,26,27,28
    blk02['PQA'] = pqa
    
    unq = buf[28:31].decode('ascii')                # 29,30,31
    blk02['UnQ'] = unq
    
    code = int.from_bytes(buf[31:32])               # 32
    blk02['Code'] = code
    
    return blk02

def read_traceheader_block3(f):
    # Measured sensor postion
    blk03 = OrderedDict()
    buf = f.read(32)
    
    x = int.from_bytes(buf[0:4], signed=True)    # 1,2,3,4
    blk03['XLoc'] = x
    
    y = int.from_bytes(buf[4:8], signed=True)   # 5,6,7,8
    blk03['YLoc'] = y
    
    z = int.from_bytes(buf[8:12], signed=True) # 9,10,11,12
    blk03['ZLoc'] = z
    
    ht = int.from_bytes(buf[12:16], signed=True)    # 13,14,15,16
    blk03['horizontal_technique'] = ht
    
    hdev = unpack('>f', buf[16:20])             # 17,18,19,20
    blk03['horizontal_deviation'] = hdev
    
    vt = int.from_bytes(buf[20:24], signed=True)    # 21,22,23,24
    blk03['vertical_technique'] = vt
    
    vdev = unpack('>f', buf[24:28])             # 25,26,27,28
    blk03['vertical_deviation'] = vdev
    
    unq = buf[28:31].decode('ascii')                # 29,30,31
    blk03['UnQ'] = unq
    
    code = int.from_bytes(buf[31:32])               # 32
    blk03['Code'] = code
    
    return blk03

def read_traceheader_block4(f):
    # Final sensor position
    blk04 = OrderedDict()
    buf = f.read(32)
    
    x = int.from_bytes(buf[0:4], signed=True)    # 1,2,3,4
    blk04['XLoc'] = x
    
    y = int.from_bytes(buf[4:8], signed=True)   # 5,6,7,8
    blk04['YLoc'] = y
    
    z = int.from_bytes(buf[8:12], signed=True) # 9,10,11,12
    blk04['ZLoc'] = z
    
    ex = int.from_bytes(buf[12:16], signed=True)    # 13,14,15,16
    blk04['LocalX'] = ex
    
    ey = int.from_bytes(buf[16:20], signed=True)    # 17,18,19.20
    blk04['LocalY'] = ey
    
    ez = int.from_bytes(buf[20:24], signed=True)    # 21,22,23,24
    blk04['LocalZ'] = ez
    
    pqa = int.from_bytes(buf[24:28], signed=True)   # 25,26,27,28
    blk04['PQA'] = pqa
    
    unq = buf[28:31].decode('ascii')                # 29,30,31
    blk04['UnQ'] = unq
    
    code = int.from_bytes(buf[31:32])               # 32
    blk04['Code'] = code
    
    return blk04

def timedriftheader(f):
    f.seek(32,1)
    """Not currently using this, come back to it later
    to see what's in here"""
    
def read_trace_header_block6(f):
    """ Misc. - not needed, come back later"""
    f.seek(32,1)
    
def read_seismic_trace_data(f, number_of_samples):
    # Verify this function somehow - produces exactly same data as Stryde MATLAB script
    data = np.zeros(number_of_samples, dtype=np.int32)  # Allocate array for the samples

    for i in range(number_of_samples):
        # Read 3 bytes (24 bits) from the file
        byte_data = f.read(3)

        # Check 3 bytes were read, position in file important
        if len(byte_data) != 3:
            raise ValueError(f"Unexpected end of file or incorrect size. Read {len(byte_data)} bytes.")

        # Convert the 3-byte data to a 32-bit signed integer
        # Sign-extend the 3 bytes to 4 bytes for interpretation as signed 32-bit integer
        if byte_data[0] & 0x80:  # Check if the sign bit is set (negative number)
            byte_data = b'\xff' + byte_data  # Add a 0xFF byte for sign-extension (big-endian)
        else:
            byte_data = b'\x00' + byte_data  # Add a 0x00 byte for positive number

        # Unpack the 4-byte data as signed 32-bit integer
        data[i] = unpack('>i', byte_data)[0]  # '>i' is big-endian 4-byte signed integer

    return data

def read_stryde(f, stryde_headers=False, 
                starttime=None,
                endtime=None,
                nearest_sample=True,
                **kwargs):
    
    f = open(f, 'rb')
    
    # General header - 3 * 32-byte blocks
    gen_hdr = read_ghb1(f)
    gen_hdr.update(read_ghb2(f))
    gen_hdr.update(read_ghb3(f))
    
    # Scan channels - 1 is always seismic data
    sch = {}
    for i in range(gen_hdr['n_channel_sets_per_scan_ghb1']):
        _sch = _read_sch(f)
        sch[_sch['channel_set_number']] = _sch
    
       
    extended_header_size = gen_hdr['extended_header_blocks_ghb2'] *32
    
    # 3rd header block is a big chunk of JSON
    # Undocumented, but contains hardware debug info and GNSS sync events
    json_header = read_json_extended_header(f, extended_header_size)
            
    # Read seismic trace headers here - wrap these in own function?
    trace_header = read_trace_header(f)
    #print("Position is:", f.tell())
    blk01 = read_traceheader_block1(f)
    #print("Reading trace extension block 1")
    trace_header['Block01'] = blk01
    #print("Position is:", f.tell())
    #print("Reading trace header block 2")
    planned = read_traceheader_block2(f)
    trace_header['Planned'] = planned
    #print("Position is:", f.tell())
    #print("Reading trace header block 3")
    measured = read_traceheader_block3(f)
    trace_header['Measured'] = measured
    #print("Position is:", f.tell())
    #print("Reading trace header block 4")
    final = read_traceheader_block4(f)
    trace_header['Final'] = final
    #print("Position is:", f.tell())
    #print("Time drift header - skip")
    timedriftheader(f)
    #print("Position in file is:", f.tell())
    #print("Skipping misc. header")
    read_trace_header_block6(f)
    #print("Position is:", f.tell())
    #print("Skipping unused trace header extensions")
    f.seek(160, 1)
    #print("Position is:", f.tell())
    #print("Reading seismic data block")
    num_samples = sch[1]['number_of_samples']
    data = read_seismic_trace_data(f, num_samples)
    descale_multiplier = sch[1]['descale_multiplier']
    mydata = data * descale_multiplier
    tr = Trace(mydata)
    tr.stats.starttime = gen_hdr['time_ghb1'] # Use leap-second corrected GPS?
    sample_rate = sch[1]['SampleInt'] / 1e6
    tr.stats.delta = sample_rate
    tr.stats.network = str(trace_header['Block01']['line_number'])
    tr.stats.station = str(trace_header['Block01']['receiver_number'])
    tr.stats.channel = 'GHZ'
    
    st = Stream()
    st.append(tr)
    
    kwargs['starttime'] = starttime
    kwargs['endtime'] = endtime
    
    if starttime:
        st._ltrim(starttime, nearest_sample=nearest_sample)
    if endtime:
        st._rtrim(endtime, nearest_sample=nearest_sample)
    
    # Return all the headers, or just the stream?
    if stryde_headers:
        return gen_hdr, sch, json_header, trace_header, st
    else:
        return st

def main():
    # Convert stryde file from command line
    par = ArgumentParser()
    par.add_argument('infile')
    par.add_argument('-o', '--output', choices=['SAC', 'MSEED'])
    
    args = par.parse_args()
    
    st = read_stryde(args.infile)
    
    if args.output == "SAC":
        outfile = f'{args.infile}.{args.output}'
    else:
        outfile = f'{args.infile}.{args.output}'
        
    st.write(outfile, format=args.output)
    
if __name__ == '__main__':
    main()
    