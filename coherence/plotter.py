
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


def showDegCoh(filename, rel_thresh=1e-4):
    print(">>>>>>")
    print("FILE: " + filename)

    coor, coor_conj, mutual_intensity = loadNumpyFormatCoh(filename)

    print("File Loaded")

    f = h5py.File("tmp0.h5",'w')
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


    f["3_x"] = nmResDegCoh.get_x_values()
    f["3_y"] = nmResDegCoh.get_y_values()
    f["3_z"] = nmResDegCoh.get_z_values()
    # f["3_mesh"] = [xmin, xmax, inx, ymin, ymax, iny]
    f.close()
    print("File tmp0.h5 written to disk.")


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

def calculate_degree_of_coherence_vs_average_and_difference(coor, coor_conj, mutual_intensity,dump_h5_file=True):

    print(coor.shape, coor_conj.shape, mutual_intensity.shape )


    #
    if dump_h5_file:
        f = h5py.File("tmp1.h5",'w')
        f["0_coor"] = coor
        f["0_coor_conj"] = coor_conj
        f["0_mutual_intensity"] = mutual_intensity

    interpolator0 = RectBivariateSpline(coor, coor_conj, mutual_intensity, bbox=[None, None, None, None], kx=3, ky=3, s=0)

    # # extending the mutual coherence (padding)
    #
    # xStart = coor[0]
    # xNp = coor.size
    # xStep = coor[1] - coor[0]
    # xEnd = xStart + (xNp - 1)*xStep
    #
    # yStart = coor_conj[0]
    # yNp =  coor_conj.size
    # yStep = coor_conj[1] - coor_conj[0]
    # yEnd = yStart + (yNp - 1)*yStep
    #
    # xNpNew = 2*xNp - 1
    # yNpNew = 2*yNp - 1
    # #
    # # print("Creating Matrix wInMutCohRes")
    #
    # xHalfNp = round(xNp*0.5)
    # yHalfNp = round(yNp*0.5)
    #
    # x0 = (xStart - xHalfNp*xStep)
    # wInMutCohRes_x = numpy.arange(x0,x0+xNpNew*xStep,xStep)
    # y0 = (yStart - yHalfNp*yStep)
    # wInMutCohRes_y = numpy.arange(y0,y0+yNpNew*yStep,yStep)
    #
    # print("Padding X to: ",wInMutCohRes_x.size,int( 0.5*(wInMutCohRes_x.size-coor.size)),  wInMutCohRes_x.size-int( 0.5*(wInMutCohRes_x.size-coor.size))-coor.size)
    # print("Padding Y to: ",wInMutCohRes_y.size,int( 0.5*(wInMutCohRes_y.size-coor.size)),  wInMutCohRes_y.size-int( 0.5*(wInMutCohRes_y.size-coor.size))-coor_conj.size)
    #
    #
    # pad = ((int( 0.5*(wInMutCohRes_x.size-coor.size)),  wInMutCohRes_x.size-int( 0.5*(wInMutCohRes_x.size-coor.size))-coor.size),
    #         (int( 0.5*(wInMutCohRes_y.size-coor.size)),  wInMutCohRes_y.size-int( 0.5*(wInMutCohRes_y.size-coor.size))-coor_conj.size))
    #
    #
    # wInMutCohRes_z = numpy.pad(mutual_intensity,pad,mode='constant')
    #
    #
    # # print("Done")
    # if dump_h5_file:
    #     f["1_x"] = wInMutCohRes_x
    #     f["1_y"] = wInMutCohRes_y
    #     f["1_z"] = wInMutCohRes_z
    #     # f["1_mesh"] = [xStart, xEnd, xNpNew, yStart, yEnd, yNpNew]
    #
    #
    # # calculate degree of coherence by interpolation TODO: really needed?
    # interpolator1 = RectBivariateSpline(wInMutCohRes_x, wInMutCohRes_y, wInMutCohRes_z, bbox=[None, None, None, None], kx=3, ky=3, s=0)
    #
    # wMutCohNonRot_x = coor
    # wMutCohNonRot_y = coor_conj
    #
    # intX = numpy.zeros_like(wMutCohNonRot_x)
    # intY = numpy.zeros_like(wMutCohNonRot_y)
    #
    # for ix,vx in enumerate(wMutCohNonRot_x):
    #     intX[ix] = interpolator1(vx,vx,grid=False)
    #
    # for iy,vy in enumerate(wMutCohNonRot_y):
    #     intY[iy] = interpolator1(vy,vy,grid=False)
    #
    # wMutCohNonRot_z = numpy.abs(interpolator1(wMutCohNonRot_x,wMutCohNonRot_y,grid=True)) / numpy.sqrt( numpy.abs(numpy.outer(intX,intY)))
    #
    #
    # print(">>>>",wMutCohNonRot_z.shape)
    # #
    # #
    # if dump_h5_file:
    #     f["2_x"] = wMutCohNonRot_x
    #     f["2_y"] = wMutCohNonRot_y
    #     f["2_z"] = wMutCohNonRot_z
    #
    # interpolator2 = RectBivariateSpline(wMutCohNonRot_x, wMutCohNonRot_y, wMutCohNonRot_z, bbox=[None, None, None, None], kx=3, ky=3, s=0)

    # calculate the "rotated" degree of coherence vs x1+x2 and x1-x2


    nmResDegCoh_x = coor
    nmResDegCoh_y = coor_conj

    X = numpy.outer(nmResDegCoh_x,numpy.ones_like(nmResDegCoh_y))
    Y = numpy.outer(numpy.ones_like(nmResDegCoh_x),nmResDegCoh_y)

    # nmResDegCoh_z = interpolator2( X+Y,X-Y, grid=False)

    nmResDegCoh_z = numpy.abs(interpolator0( X+Y,X-Y, grid=False)) /\
                numpy.sqrt(numpy.abs(interpolator0( X+Y,X+Y, grid=False))) /\
                numpy.sqrt(numpy.abs(interpolator0( X-Y,X-Y, grid=False)))

    if dump_h5_file:
        f["3_x"] = nmResDegCoh_x
        f["3_y"] = nmResDegCoh_y
        f["3_z"] = nmResDegCoh_z

        f.close()
        print("File tmp1.h5 written to disk.")

    return nmResDegCoh_x,nmResDegCoh_y,nmResDegCoh_z


def showDegCoh_srio(filename, rel_thresh=1e-4, direction='y',do_plot=True):

    coor, coor_conj, mutual_intensity  = srio_get_from_h5_file(filename,"0_coor", "0_coor_conj", "0_mutual_intensity")

    x, y, z = calculate_degree_of_coherence_vs_average_and_difference(coor,coor_conj,mutual_intensity)

    if do_plot:
        if direction == 'x':
            xlabel = "(x1+x2)/2 [um]"
            ylabel = "(x1-x2)/2 [um]"
        elif direction == 'y':
            xlabel = "(y1+y2)/2 [um]"
            ylabel = "(y1-y2)/2 [um]"

        plot_scaled_matrix_srio(z, x, y, "nmResDegCoh", xlabel, ylabel)

def compare_files(filenew,fileold):
    from numpy.testing import assert_almost_equal
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

    # this is the old calculation - redo it once for creatinh tmp0.h5
    # filename = "/users/srio/OASYS1/srw-scripts/coherence/esrf_TE_50k_d0_ME_AP_CrossSpectralDensity_vertical_cut_noErr.dat" # sys.argv[1]
    # showDegCoh(filename)

    # if filename.endswith(".dat"):
    #     showPlot(filename)
    # else:
    #     showDegCoh(filename)

    showDegCoh_srio("tmp0.h5",direction='y',do_plot=True)
    compare_files("tmp1.h5","tmp0.h5")
    # plot_from_file("tmp0.h5")
    app.exec()

