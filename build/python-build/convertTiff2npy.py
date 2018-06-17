import numpy
from osgeo import gdal
import osgeo.gdalnumeric as gdn

def img_to_array(input_file, dim_ordering="channels_last", dtype='float32'):
	file  = gdal.Open(input_file)
	bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
	arr = numpy.array([gdn.BandReadAsArray(band) for band in bands])
	arr = arr.astype(dtype)
	if dim_ordering=="channels_last":
		arr = numpy.transpose(arr, [1, 2, 0])  # Reorders dimensions, so that channels are last
	return arr


import os
import sys
if len(sys.argv)>1 :
	file_name = sys.argv[1]
	image=img_to_array(file_name)
	numpy.save(os.path.splitext(file_name)[0],image)