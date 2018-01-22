#!/usr/bin/env python
'''Writes a NeXus HDF5 file using h5py and numpy'''

import h5py    # HDF5 support
import numpy
import sys

print("Write a NeXus HDF5 file")

if sys.version < "3":
    fileName = "nxdata2d.nexus.hdf5"
else:
    fileName = "nxdata2d.nexus_py3.hdf5"
timestamp = "2010-10-18T17:17:04-0500"

# load data from two column format
data = numpy.arange(100000.)
data.shape = 100, 1000

# create the HDF5 NeXus file
f = h5py.File(fileName, "w")
# point to the default data to be plotted
f.attrs['default']          = 'entry'
# give the HDF5 root some more attributes
f.attrs['file_name']        = fileName
f.attrs['file_time']        = timestamp
f.attrs['creator']          = 'NXdataImage.py'
f.attrs['HDF5_Version']     = h5py.version.hdf5_version
f.attrs['h5py_version']     = h5py.version.version

# create the NXentry group
nxentry = f.create_group('entry')
nxentry.attrs['NX_class'] = 'NXentry'
nxentry.attrs['default'] = 'image_plot'
nxentry.create_dataset('title', data='2D Image')
# nxentry = f

# create the NXentry group
nxdata = nxentry.create_group('image_plot')
nxdata.attrs['NX_class'] = 'NXdata'
nxdata.attrs['signal'] = 'image_data'              # Y axis of default plot
nxdata.attrs['axes'] = [b'axis0_name', b'axis1_name'] # X axis of default plot

# signal data
ds = nxdata.create_dataset('image_data', data=data)
ds.attrs['interpretation'] = 'image'

# X axis data
ds = nxdata.create_dataset('axis1_name', data=numpy.arange(data.shape[1])-0.5*int(data.shape[1]))
ds.attrs['units'] = 'microns'
ds.attrs['long_name'] = 'Pixel Size (microns)'    # suggested X axis plot label

# Y axis data
ds = nxdata.create_dataset('axis0_name', data=numpy.arange(data.shape[0])-0.5*int(data.shape[0]))
ds.attrs['units'] = 'microns'
ds.attrs['long_name'] = 'Pixel Size (microns)'    # suggested Y axis plot label

f.close()   # be CERTAIN to close the file

print("wrote file:", fileName)
