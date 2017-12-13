
import numpy
import sys

from srxraylib.util.data_structures import ScaledMatrix, ScaledArray
from silx.gui.plot import Plot2D, ImageView

import h5py

from scipy.interpolate import RectBivariateSpline
from PyQt5.QtWidgets import QApplication

# try:
#     import matplotlib.pylab as plt
#     plt.switch_backend("Qt5Agg")
# except:
#     raise Exception("Failed to set matplotlib backend to Qt5Agg")

# copied from SRW's uti_plot_com and slightly  modified (no _enum)
def file_load(_fname, _read_labels=1):
    import numpy as np
    nLinesHead = 11
    hlp = []

    with open(_fname,'r') as f:
        for i in range(nLinesHead):
            hlp.append(f.readline())

    ne, nx, ny = [int(hlp[i].replace('#','').split()[0]) for i in [3,6,9]]
    ns = 1
    testStr = hlp[nLinesHead - 1]
    if testStr[0] == '#':
        ns = int(testStr.replace('#','').split()[0])

    e0,e1,x0,x1,y0,y1 = [float(hlp[i].replace('#','').split()[0]) for i in [1,2,4,5,7,8]]

    data = numpy.squeeze(numpy.loadtxt(_fname, dtype=numpy.float64)) #get data from file (C-aligned flat)

    allrange = e0, e1, ne, x0, x1, nx, y0, y1, ny

    arLabels = ['Photon Energy', 'Horizontal Position', 'Vertical Position', 'Intensity']
    arUnits = ['eV', 'm', 'm', 'ph/s/.1%bw/mm^2']

    if _read_labels:

        arTokens = hlp[0].split(' [')
        arLabels[3] = arTokens[0].replace('#','')
        arUnits[3] = '';
        if len(arTokens) > 1:
            arUnits[3] = arTokens[1].split('] ')[0]

        for i in range(3):
            arTokens = hlp[i*3 + 1].split()
            nTokens = len(arTokens)
            nTokensLabel = nTokens - 3
            nTokensLabel_mi_1 = nTokensLabel - 1
            strLabel = ''
            for j in range(nTokensLabel):
                strLabel += arTokens[j + 2]
                if j < nTokensLabel_mi_1: strLabel += ' '
            arLabels[i] = strLabel
            arUnits[i] = arTokens[nTokens - 1].replace('[','').replace(']','')

    return data, None, allrange, arLabels, arUnits

def loadNumpyFormat(filename):
    data, dump, allrange, arLabels, arUnits = file_load(filename)

    dim_x = allrange[5]
    dim_y = allrange[8]
    np_array = data.reshape((dim_y, dim_x))
    np_array = np_array.transpose()
    x_coordinates = numpy.linspace(allrange[3], allrange[4], dim_x)
    y_coordinates = numpy.linspace(allrange[6], allrange[7], dim_y)

    return x_coordinates, y_coordinates, np_array

def loadNumpyFormatCoh(filename):
    data, dump, allrange, arLabels, arUnits = file_load(filename)

    dim_x = allrange[5]
    dim_y = allrange[8]

    dim = 1
    if dim_x > 1: dim = dim_x
    elif dim_y > 1: dim = dim_y

    np_array = data.reshape((dim, dim))
    np_array = np_array.transpose()

    if dim_x > 1:
        coordinates = numpy.linspace(allrange[3], allrange[4], dim_x)
        conj_coordinates = numpy.linspace(allrange[3], allrange[4], dim_x)
    elif dim_y > 1:
        coordinates = numpy.linspace(allrange[6], allrange[7], dim_y)
        conj_coordinates = numpy.linspace(allrange[6], allrange[7], dim_y)
    else:
        coordinates = None
        conj_coordinates = None

    return coordinates, conj_coordinates, np_array

def plot_2D(x, y, data, title, xlabel="X [um]", ylabel="Y [um]"):
    xmin, xmax = x.min()*1e6, x.max()*1e6
    ymin, ymax = y.min()*1e6, y.max()*1e6

    origin = (xmin, ymin)
    scale = (abs((xmax-xmin)/len(x)), abs((ymax-ymin)/len(y)))

    # PyMCA inverts axis!!!! histogram must be calculated reversed
    data_to_plot = []
    for y_index in range(0, len(y)):
        x_values = []
        for x_index in range(0, len(x)):
            x_values.append(data[x_index, y_index])

        x_values.reverse()

        data_to_plot.append(x_values)

    plot_canvas = ImageView()

    #plot_canvas = Plot2D()
    plot_canvas.setGraphTitle(title)
    plot_canvas.addImage(numpy.array(data_to_plot),
                         xlabel=xlabel,
                         ylabel=ylabel,
                         legend="data",
                         origin=origin,
                         scale=scale,
                         replace=True)

    plot_canvas.toolBar()
    plot_canvas.show()
    
def plot_scaled_matrix(scaled_matrix, title, xlabel="X [um]", ylabel="Y [um]"):
    plot_2D(scaled_matrix.get_x_values(), scaled_matrix.get_y_values(), scaled_matrix.get_z_values(), title, xlabel, ylabel)

def plot_scaled_matrix_srio(z, x, y, title, xlabel="X [um]", ylabel="Y [um]"):
    # from srxraylib.plot.gol import plot_image
    # print(">>>>>>>>>>>",z.shape,x.shape,y.shape)
    # plot_image(z, x, y, title=title, xtitle=xlabel, ytitle=ylabel)
    plot_2D(x, y, z, title, xlabel, ylabel)


def srwUtiNonZeroIntervB(p, pmin, pmax):
    if((p < pmin) or (p > pmax)):
        return 0.
    else:
        return 1.

def srwUtiInterp2DBilin(x, y, matrix, xmin, xmax, xstep, ymin, ymax, ystep):

    if((x < xmin) or (x > xmax) or (y < ymin) or (y > ymax)):
        return 0

    x0 = xmin + numpy.trunc((x - xmin)/xstep)*xstep
    if(x0 >= xmax): x0 = xmax - xstep

    x1 = x0 + xstep

    y0 = ymin + numpy.trunc((y - ymin)/ystep)*ystep
    if(y0 >= ymax): y0 = ymax - ystep

    y1 = y0 + ystep

    t = (x - x0)/xstep
    u = (y - y0)/ystep

    return (1 - t)*(1 - u)*(matrix.interpolate_value(x0, y0)) + \
           t*(1 - u)*(matrix.interpolate_value(x1, y0)) + \
           t*u*(matrix.interpolate_value(x1, y1)) + \
           (1 - t)*u*(matrix.interpolate_value(x0, y1))

def showPlot(filename):
    print(">>>>>>")
    print("FILE: " + filename)

    coor, coor_conj, inten = loadNumpyFormat(filename)

    plot_2D(coor, coor_conj, inten, filename)


def showDegCoh(filename, file_h5=None, rel_thresh=1e-4):
    print(">>>>>>")
    print("FILE: " + filename)

    coor, coor_conj, mutual_intensity = loadNumpyFormatCoh(filename)

    print("File Loaded")

    if file_h5 is not None:
        f = h5py.File(file_h5,'w')
        f["0_coor"] = coor
        f["0_coor_conj"] = coor_conj
        f["0_mutual_intensity"] = mutual_intensity
        # f["0_mesh"] = [xStart, xEnd, xNpNew, yStart, yEnd, yNpNew]

    #-----------------------------------------------------------
    #FROM OLEG'S IGOR MACRO ------------------------------------
    #-----------------------------------------------------------

    nmInMutInt = ScaledMatrix(x_coord=coor, y_coord=coor_conj, z_values=mutual_intensity, interpolator=True)

    xStart = nmInMutInt.offset_x()
    xNp = nmInMutInt.size_x()
    xStep = nmInMutInt.delta_x()
    xEnd = xStart + (xNp - 1)*xStep

    yStart = nmInMutInt.offset_y()
    yNp =  nmInMutInt.size_y()
    yStep = nmInMutInt.delta_y()
    yEnd = yStart + (yNp - 1)*yStep

    xNpNew = 2*xNp - 1
    yNpNew = 2*yNp - 1

    print("Creating Matrix wInMutCohRes")

    wInMutCohRes = ScaledMatrix(x_coord=numpy.zeros(xNpNew),
                                y_coord=numpy.zeros(yNpNew),
                                z_values=numpy.zeros((xNpNew, yNpNew)),
                                interpolator=False)

    xHalfNp = round(xNp*0.5)
    yHalfNp = round(yNp*0.5)

    wInMutCohRes.set_scale_from_steps(axis=0, initial_scale_value=(xStart - xHalfNp*xStep), scale_step=xStep)
    wInMutCohRes.set_scale_from_steps(axis=1, initial_scale_value=(yStart - yHalfNp*yStep), scale_step=yStep)

    dimx, dimy = wInMutCohRes.shape()
    for inx in range(0, dimx):
        for iny in range(0,dimy):
            x = wInMutCohRes.get_x_value(inx)
            y = wInMutCohRes.get_y_value(iny)

            wInMutCohRes.set_z_value(inx, iny,
                                     nmInMutInt.interpolate_value(x, y)*
                                     srwUtiNonZeroIntervB(x,
                                                          xStart,
                                                          xEnd)*
                                     srwUtiNonZeroIntervB(y,
                                                          yStart,
                                                          yEnd))

    wInMutCohRes.compute_interpolator()

    print("Done")


    if file_h5 is not None:
        f["1_x"] = wInMutCohRes.get_x_values()
        f["1_y"] = wInMutCohRes.get_y_values()
        f["1_z"] = wInMutCohRes.get_z_values()
        # f["1_mesh"] = [xStart, xEnd, xNpNew, yStart, yEnd, yNpNew]


    print("Creating Matrix wMutCohNonRot")

    wMutCohNonRot = ScaledMatrix(x_coord=nmInMutInt.get_x_values(),
                                 y_coord=nmInMutInt.get_y_values(),
                                 z_values=numpy.zeros(nmInMutInt.shape()),
                                 interpolator=False)

    abs_thresh = rel_thresh*abs(nmInMutInt.interpolate_value(0, 0))

    dimx, dimy = wMutCohNonRot.shape()
    for inx in range(0, dimx):
        for iny in range(0, dimy):
            x = wMutCohNonRot.get_x_value(inx)
            y = wMutCohNonRot.get_y_value(iny)

            wMutCohNonRot.set_z_value(inx, iny,
                                      numpy.abs(wInMutCohRes.interpolate_value(x, y))/
                                      (numpy.sqrt(abs(wInMutCohRes.interpolate_value(x, x)*wInMutCohRes.interpolate_value(y, y))) + abs_thresh))


    wMutCohNonRot.compute_interpolator()

    print("Done")

    if file_h5 is not None:
        f["2_x"] = wMutCohNonRot.get_x_values()
        f["2_y"] = wMutCohNonRot.get_y_values()
        f["2_z"] = wMutCohNonRot.get_z_values()




    print("Creating Matrix nmResDegCoh")

    nmResDegCoh = ScaledMatrix(x_coord=nmInMutInt.get_x_values(),
                               y_coord=nmInMutInt.get_y_values(),
                               z_values=numpy.zeros(nmInMutInt.shape()),
                               interpolator=False)

    xmin = wMutCohNonRot.offset_x()
    nx = wMutCohNonRot.size_x()
    xstep = wMutCohNonRot.delta_x()
    xmax = xmin + (nx - 1)*xstep

    ymin = wMutCohNonRot.offset_y()
    ny = wMutCohNonRot.size_y()
    ystep = wMutCohNonRot.delta_y()
    ymax = ymin + (ny - 1)*ystep


    dimx, dimy = nmResDegCoh.shape()
    for inx in range(0, dimx):
        for iny in range(0, dimy):
            x = nmResDegCoh.get_x_value(inx)
            y = nmResDegCoh.get_y_value(iny)

            nmResDegCoh.set_z_value(inx, iny, srwUtiInterp2DBilin((x+y),
                                                                  (x-y),
                                                                  wMutCohNonRot,
                                                                  xmin,
                                                                  xmax,
                                                                  xstep,
                                                                  ymin,
                                                                  ymax,
                                                                  ystep))

    print("Done: plotting Results")


    if file_h5 is not None:
        f["3_x"] = nmResDegCoh.get_x_values()
        f["3_y"] = nmResDegCoh.get_y_values()
        f["3_z"] = nmResDegCoh.get_z_values()
        # f["3_mesh"] = [xmin, xmax, inx, ymin, ymax, iny]
        f.close()
        print("File %s written to disk."%file_h5)


    if filename.endswith("1"):
        xlabel = "(x1+x2)/2 [um]"
        ylabel = "(x1-x2)/2 [um]"
    else:
        xlabel = "(y1+y2)/2 [um]"
        ylabel = "(y1-y2)/2 [um]"

    plot_scaled_matrix(nmResDegCoh, "nmResDegCoh", xlabel, ylabel)


#
# new routines...
#

def calculate_degree_of_coherence_vs_sum_and_difference(coor, coor_conj, mutual_intensity, set_extrapolated_to_zero=True, filename=None):
    """
    Calculates the modulus of the complex degree of coherence versus coordinates x1+x2 and x1-x2
        (or y1+y2 and y1-y2)

    :param coor: the x1 or y1 coordinate
    :param coor_conj: the x2 or y2 coordinate
    :param mutual_intensity: the mutual intensity vs (x1,x2) [or y2,y3]
    :param filename: Name of output hdf5 filename (optional, default=None, no output file)
    :return: x1,x2,DOC
    """

    #
    if filename is not None:
        f = h5py.File(filename,'w')
        f["0_coor"] = coor
        f["0_coor_conj"] = coor_conj
        f["0_mutual_intensity"] = mutual_intensity

    interpolator0 = RectBivariateSpline(coor, coor_conj, mutual_intensity, bbox=[None, None, None, None], kx=3, ky=3, s=0)

    X = numpy.outer(coor,numpy.ones_like(coor_conj))
    Y = numpy.outer(numpy.ones_like(coor),coor_conj)

    nmResDegCoh_z = numpy.abs(interpolator0( X+Y,X-Y, grid=False)) /\
                numpy.sqrt(numpy.abs(interpolator0( X+Y,X+Y, grid=False))) /\
                numpy.sqrt(numpy.abs(interpolator0( X-Y,X-Y, grid=False)))

    if set_extrapolated_to_zero:
        nx,ny = nmResDegCoh_z.shape

        idx = numpy.outer(numpy.arange(nx),numpy.ones((ny)))
        idy = numpy.outer(numpy.ones((nx)),numpy.arange(ny))

        mask = numpy.ones_like(idx)

        bad = numpy.where(idy < 1.*(idx-nx/2)*ny/nx)
        mask[ bad ] = 0

        bad = numpy.where(idy > ny - 1.*(idx-nx/2)*ny/nx)
        mask[ bad ] = 0

        bad = numpy.where(idy < 0.5*ny - 1.*idx*ny/nx)
        mask[ bad ] = 0

        bad = numpy.where(idy > 0.5*ny + 1.*idx*ny/nx)
        mask[ bad ] = 0

        nmResDegCoh_z *= mask


    if filename is not None:
        f["3_x"] = coor
        f["3_y"] = coor_conj
        f["3_z"] = nmResDegCoh_z

        f.close()
        print("File %s written to disk."%filename)

    return coor,coor_conj,nmResDegCoh_z


def showDegCoh_srio(filename_in, filename_out, rel_thresh=1e-4, direction='y',do_plot=True):

    coor, coor_conj, mutual_intensity  = srio_get_from_h5_file(filename_in,"0_coor", "0_coor_conj", "0_mutual_intensity")

    x, y, z = calculate_degree_of_coherence_vs_sum_and_difference(coor,coor_conj,mutual_intensity,filename=filename_out)

    if do_plot:
        if direction == 'x':
            xlabel = "(x1+x2) [um]"
            ylabel = "(x1-x2) [um]"
        elif direction == 'y':
            xlabel = "(y1+y2) [um]"
            ylabel = "(y1-y2) [um]"

        plot_scaled_matrix_srio(z, x, y, "nmResDegCoh", xlabel, ylabel)


def coherent_length_from_file(filein):

    x,y,z = srio_get_from_h5_file(filein,"3_x","3_y","3_z")
    #
    # calculate coherent length
    #
    nx = z.shape[0]
    h = z[int(nx/2),:]
    tt = numpy.where(h>=max(h)*0.5)
    tt0 = 0
    tt1 = -1
    if h[tt].size > 1:
      binSize = y[1]-y[0]
      print('fwhm', binSize*(tt[tt0][tt1]-tt[tt0][tt0]))
      print('fwhm_coordinates', (y[tt[tt0][tt0]],y[tt[tt0][tt1]]))

    # from srxraylib.plot.gol import plot
    # plot(y,h,show=1)
    # plot(y,numpy.gradient(h),title="derivative",show=0)
    # plot(y,numpy.gradient(numpy.gradient(h)),title="second derivative")


def compare_files(filenew,fileold):
    # from numpy.testing import assert_almost_equal
    fnew = h5py.File(filenew,'r')
    fold = h5py.File(fileold,'r')

    for key in fnew.keys():
        print(key,fnew[key].shape,fold[key].shape)
        # assert_almost_equal(fnew[key].value,fold[key].value,3)
        print("   ",fnew[key].value[1024],fold[key].value[1024])

    fnew.close()
    fold.close()

def srio_get_from_h5_file(filename,a1=None,a2=None,a3=None):
    f = h5py.File(filename,'r')

    out = []
    if a1 is not None:
        out.append(f[a1].value)
    if a2 is not None:
        out.append(f[a2].value)
    if a3 is not None:
        out.append(f[a3].value)
    return out

def plot_from_file(filename):
    f = h5py.File(filename,'r')
    x = f["3_x"].value
    y = f["3_y"].value
    z = f["3_z"].value
    f.close()

    plot_scaled_matrix_srio(z, x, y, filename, "1+2", "1-2")

if __name__== "__main__":

    app = QApplication(sys.argv)

    #
    # this is the old calculation - redo it once for creating tmp0.h5 that will be used for data input and comparison of results
    #

    # filename = "/users/srio/OASYS1/srw-scripts/coherence/esrf_TE_50k_d0_ME_AP_CrossSpectralDensity_vertical_cut_noErr.dat" # sys.argv[1]
    # showDegCoh(filename,file_h5="tmp0.h5")

    # filename = "/tmp_14_days/celestre/cross_spectral_files/ebs_TE_50k_d0_ME_AP_CrossSpectralDensity_horizontal_noErr.dat"
    # showDegCoh(filename,file_h5="tmp00.h5")

    # if filename.endswith(".dat"):
    #     showPlot(filename)
    # else:
    #     showDegCoh(filename)

    #
    # new calculation
    #


    showDegCoh_srio(filename_in="tmp00.h5",filename_out="tmp11.h5",direction='y',do_plot=True)
    # compare_files("tmp1.h5","tmp0.h5")
    # plot_from_file("tmp00.h5")
    # coherent_length_from_file("tmp00.h5")
    # app.exec()

