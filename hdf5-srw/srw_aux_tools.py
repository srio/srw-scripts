from __future__ import print_function #Python 2.7 compatibility
from srwlib import *
from uti_plot import *
from math import *
import os
import copy
import numpy as np

from srxraylib.plot.gol import plot,plot_image
import numpy
import sys
import h5py

import scipy.constants as codata

try:
    import h5py
    has_h5py = True
except:
    print("h5py is not installed")
    has_h5py = False


#
#**********************Auxiliary functions to calculate execution time
#
def tic():
    import time
    global startTime
    startTime = time.time()


def toc(text=""):
    import time
    print('')
    if 'startTime' in globals():
        deltaT = time.time() - startTime
        hours, minutes = divmod(deltaT, 3600)
        minutes, seconds =  divmod(minutes, 60)
        print("Elapsed time "+text+" : " + str(int(hours)) + "h " + str(int(minutes)) + "min " + str(seconds) + "s ")
    else:
        print("Warning: start time not set.")


#
# srio tools
#
def plot_wfr(wfr,kind='intensity',show=True,xtitle="X",ytitle="Y",title="",aspect='auto'):

	if kind == 'intensity':
		ar1 = array('f', [0]*wfr.mesh.nx*wfr.mesh.ny) # "flat" 2D array to take intensity data
		srwl.CalcIntFromElecField(ar1, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
	elif kind == 'phase':
		ar1 = array('d', [0]*wfr.mesh.nx*wfr.mesh.ny) # "flat" array to take 2D phase data (note it should be 'd')
		srwl.CalcIntFromElecField(ar1, wfr, 0, 4, 3, wfr.mesh.eStart, 0, 0)
	else:
		raise Exception("Unknown kind of calculation: %s"%(kind))

	arxx = numpy.array(ar1)
	arxx = arxx.reshape((wfr.mesh.ny,wfr.mesh.nx)).T
	x = numpy.linspace(1e6*wfr.mesh.xStart, 1e6*wfr.mesh.xFin, wfr.mesh.nx)
	y = numpy.linspace(1e6*wfr.mesh.yStart, 1e6*wfr.mesh.yFin, wfr.mesh.ny)

	plot_image(arxx, x, y, xtitle="%s (%d pixels)"%(xtitle,x.size), ytitle="%s (%d pixels)"%(ytitle,y.size), title=title, aspect=aspect,show=show)

	return ar1,x,y

def dump_open(filename="tmp.h5"):
	sys.stdout.flush()

	f = h5py.File(filename, 'w')

	return f

def dump_wfr(f,wfr,prefix="",phase=False,complex_amplitude=False):

    x = numpy.linspace(1e6*wfr.mesh.xStart, 1e6*wfr.mesh.xFin, wfr.mesh.nx)
    y = numpy.linspace(1e6*wfr.mesh.yStart, 1e6*wfr.mesh.yFin, wfr.mesh.ny)

    f[prefix+"_x"] = x
    f[prefix+"_y"] = y

    ar1 = array('f', [0]*wfr.mesh.nx*wfr.mesh.ny) # "flat" 2D array to take intensity data
    srwl.CalcIntFromElecField(ar1, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
    arxx = numpy.array(ar1)
    arxx = arxx.reshape((wfr.mesh.ny,wfr.mesh.nx)).T
    f[prefix+"_intensity"] = arxx

    if phase:
        ar1 = array('d', [0]*wfr.mesh.nx*wfr.mesh.ny) # "flat" array to take 2D phase data (note it should be 'd')
        srwl.CalcIntFromElecField(ar1, wfr, 0, 4, 3, wfr.mesh.eStart, 0, 0)
        arxx = numpy.array(ar1)
        arxx = arxx.reshape((wfr.mesh.ny,wfr.mesh.nx)).T
        f[prefix+"_phase"] = arxx

	# complex amplitude

    if complex_amplitude:
        dim_x = wfr.mesh.nx
        dim_y = wfr.mesh.ny
        number_energies = wfr.mesh.ne

        x_polarization = SRWArrayToNumpy(wfr.arEx, dim_x, dim_y, number_energies)
        y_polarization = SRWArrayToNumpy(wfr.arEy, dim_x, dim_y, number_energies)

        e_field = numpy.concatenate((x_polarization, y_polarization), 3)

        f[prefix+"_complexamplitude_sigma"] = e_field[0,:,:,0]
        f[prefix+"_complexamplitude_pi"] = e_field[0,:,:,1]




def dump_close(f):
	f.close()

def SRWArrayToNumpy(srw_array, dim_x, dim_y, number_energies):
    """
    Converts a SRW array to a numpy.array.
    :param srw_array: SRW array
    :param dim_x: size of horizontal dimension
    :param dim_y: size of vertical dimension
    :param number_energies: Size of energy dimension
    :return: 4D numpy array: [energy, horizontal, vertical, polarisation={0:horizontal, 1: vertical}]
    """
    re = numpy.array(srw_array[::2], dtype=numpy.float)
    im = numpy.array(srw_array[1::2], dtype=numpy.float)

    e = re + 1j * im
    e = e.reshape((dim_y,
                   dim_x,
                   number_energies,
                   1)
                  )

    e = e.swapaxes(0, 2)

    return e.copy()



#
# modified by srio to save some intermediate information
#



#**********************Main Partially-Coherent Emission and Propagaiton simulation function
def srwl_wfr_emit_prop_multi_e_NEW(_e_beam, _mag, _mesh, _sr_meth, _sr_rel_prec, _n_part_tot, _n_part_avg_proc=1, _n_save_per=100,
                               _file_path=None, _sr_samp_fact=-1, _opt_bl=None, _pres_ang=0, _char=0, _x0=0, _y0=0, _e_ph_integ=0,
                               _rand_meth=1, _tryToUseMPI=True, _wr=0., filename=None,
                                   save_individual_electrons=False):
    """
    Calculate Stokes Parameters of Emitted (and Propagated, if beamline is defined) Partially-Coherent SR
    :param _e_beam: Finite-Emittance e-beam (SRWLPartBeam type)
    :param _mag: Magnetic Field container (magFldCnt type)
    :param _mesh: mesh vs photon energy, horizontal and vertical positions (SRWLRadMesh type) on which initial SR should be calculated
    :param _sr_meth: SR Electric Field calculation method to be used (0- "manual", 1- "auto-undulator", 2- "auto-wiggler")
    :param _sr_rel_prec: relative precision for SR Electric Field calculation (usually 0.01 is OK, the smaller the more accurate)
    :param _n_part_tot: total number of "macro-electrons" to be used in the calculation
    :param _n_part_avg_proc: number of "macro-electrons" to be used in calculation at each "slave" before sending Stokes data to "master" (effective if the calculation is run via MPI)
    :param _n_save_per: periodicity of saving intermediate average Stokes data to file by master process
    :param _file_path: path to file for saving intermediate average Stokes data by master process
    :param _sr_samp_fact: oversampling factor for calculating of initial wavefront for subsequent propagation (effective if >0)
    :param _opt_bl: optical beamline (container) to propagate the radiation through (SRWLOptC type)
    :param _pres_ang: switch specifying presentation of the resulting Stokes parameters: coordinate (0) or angular (1)
    :param _char: radiation characteristic to calculate:
        0- Intensity (s0);
        1- Four Stokes components;
        2- Mutual Intensity Cut vs X;
        3- Mutual Intensity Cut vs Y;
        4- Mutual Intensity Cut vs X & Y;
        10- Flux
    :param _x0: horizontal center position for mutual intensity calculation
    :param _y0: vertical center position for mutual intensity calculation
    :param _e_ph_integ: integration over photon energy is required (1) or not (0); if the integration is required, the limits are taken from _mesh
    :param _rand_meth: method for generation of pseudo-random numbers for e-beam phase-space integration:
        1- standard pseudo-random number generator
        2- Halton sequences
        3- LPtau sequences (to be implemented)
    :param _tryToUseMPI: switch specifying whether MPI should be attempted to be used
    :param _wr: initial wavefront radius [m] to assume at wavefront propagation (is taken into account if != 0)
    """


    nProc = 1
    rank = 1
    MPI = None
    comMPI = None

    if filename is not None:
        fn = dump_open(filename)
        print(">> Dumping to file %s"%filename)



    if(_tryToUseMPI):
        try:

            from mpi4py import MPI #OC091014

            comMPI = MPI.COMM_WORLD
            rank = comMPI.Get_rank()
            nProc = comMPI.Get_size()

        except:
            print('Calculation will be sequential (non-parallel), because "mpi4py" module can not be loaded')


    wfr = SRWLWfr() #Wavefronts to be used in each process
    wfr.allocate(_mesh.ne, _mesh.nx, _mesh.ny) #Numbers of points vs Photon Energy, Horizontal and Vertical Positions
    wfr.mesh.set_from_other(_mesh)
    wfr.partBeam = deepcopy(_e_beam)
    #arPrecParSR = [_sr_meth, _sr_rel_prec, 0, 0, 50000, 0, _sr_samp_fact] #to add npTraj, useTermin ([4], [5]) terms as input parameters
    arPrecParSR = [_sr_meth, _sr_rel_prec, 0, 0, 50000, 1, _sr_samp_fact] #to add npTraj, useTermin ([4], [5]) terms as input parameters

    #meshRes = SRWLRadMesh()
    meshRes = SRWLRadMesh(_mesh.eStart, _mesh.eFin, _mesh.ne, _mesh.xStart, _mesh.xFin, _mesh.nx, _mesh.yStart, _mesh.yFin, _mesh.ny, _mesh.zStart) #to ensure correct final mesh if _opt_bl==None

    ePhIntegMult = 1
    if(_e_ph_integ == 1): #Integrate over Photon Energy
        eAvg = 0.5*(_mesh.eStart + _mesh.eFin)
        ePhIntegMult = 1000*(_mesh.eFin - _mesh.eStart)/eAvg #To obtain photon energy integrated Intensity in [ph/s/mm^2] assuming monochromatic Spectral Intensity in [ph/s/.1%bw/mm^2]
        wfr.mesh.eStart = eAvg
        wfr.mesh.eFin = eAvg
        wfr.mesh.ne = 1
        meshRes.eStart = eAvg
        meshRes.eFin = eAvg
        meshRes.ne = 1

    calcSpecFluxSrc = False
    if((_char == 10) and (_mesh.nx == 1) and (_mesh.ny == 1)):
        calcSpecFluxSrc = True
        ePhIntegMult *= 1.e+06*(_mesh.xFin - _mesh.xStart)*(_mesh.yFin - _mesh.yStart) #to obtain Flux from Intensity (Flux/mm^2)

    elecX0 = _e_beam.partStatMom1.x
    elecXp0 = _e_beam.partStatMom1.xp
    elecY0 = _e_beam.partStatMom1.y
    elecYp0 = _e_beam.partStatMom1.yp
    elecGamma0 = _e_beam.partStatMom1.gamma
    elecE0 = elecGamma0*(0.51099890221e-03) #Assuming electrons

    elecSigXe2 = _e_beam.arStatMom2[0] #<(x-x0)^2>
    elecMXXp = _e_beam.arStatMom2[1] #<(x-x0)*(xp-xp0)>
    elecSigXpe2 = _e_beam.arStatMom2[2] #<(xp-xp0)^2>
    elecSigYe2 =_e_beam.arStatMom2[3] #<(y-y0)^2>
    elecMYYp = _e_beam.arStatMom2[4] #<(y-y0)*(yp-yp0)>
    elecSigYpe2 = _e_beam.arStatMom2[5] #<(yp-yp0)^2>
    elecRelEnSpr = sqrt(_e_beam.arStatMom2[10]) #<(E-E0)^2>/E0^2
    elecAbsEnSpr = elecE0*elecRelEnSpr
    #print('DEBUG MESSAGE: elecAbsEnSpr=', elecAbsEnSpr)

    multX = 0.5/(elecSigXe2*elecSigXpe2 - elecMXXp*elecMXXp)
    BX = elecSigXe2*multX
    GX = elecSigXpe2*multX
    AX = elecMXXp*multX
    SigPX = 1/sqrt(2*GX)
    SigQX = sqrt(GX/(2*(BX*GX - AX*AX)))
    multY = 0.5/(elecSigYe2*elecSigYpe2 - elecMYYp*elecMYYp)
    BY = elecSigYe2*multY
    GY = elecSigYpe2*multY
    AY = elecMYYp*multY
    SigPY = 1/sqrt(2*GY)
    SigQY = sqrt(GY/(2*(BY*GY - AY*AY)))

    #_sr_rel_prec = int(_sr_rel_prec)

    _n_part_tot = int(_n_part_tot)
    _n_part_avg_proc = int(_n_part_avg_proc)
    if(_n_part_avg_proc <= 0): _n_part_avg_proc = 1
    _n_save_per = int(_n_save_per)

    nPartPerProc = _n_part_tot
    nSentPerProc = 0

    if(nProc <= 1):
        _n_part_avg_proc = _n_part_tot
    else: #OC050214: adjustment of all numbers of points, to make sure that sending and receiving are consistent

        nPartPerProc = int(round(_n_part_tot/(nProc - 1)))
        nSentPerProc = int(round(nPartPerProc/_n_part_avg_proc)) #Number of sending acts made by each worker process

        if(nSentPerProc <= 0): #OC160116
            nSentPerProc = 1
            _n_part_avg_proc = nPartPerProc

        nPartPerProc = _n_part_avg_proc*nSentPerProc #Number of electrons treated by each worker process

    #print('DEBUG MESSAGE: rank:', rank,': nPartPerProc=', nPartPerProc, 'nSentPerProc=', nSentPerProc, '_n_part_avg_proc=', _n_part_avg_proc)


    phase_space_electrons = numpy.zeros((6,nPartPerProc))

    useGsnBmSrc = False
    if(isinstance(_mag, SRWLGsnBm)):
        useGsnBmSrc = True
        arPrecParSR = [_sr_samp_fact]
        _mag = deepcopy(_mag)
        _mag.x = elecX0
        _mag.xp = elecXp0
        _mag.y = elecY0
        _mag.yp = elecYp0
        #print('Gaussian Beam')
        #sys.exit()

    resStokes = None
    workStokes = None
    iAvgProc = 0
    iSave = 0

    doMutual = 0
    if((_char >= 2) and (_char <= 4)): doMutual = 1

    resEntityName = 'Intensity' #OC26042016
    resEntityUnits = 'ph/s/.1%bw/mm^2'
    if(calcSpecFluxSrc == True):
        resEntityName = 'Flux'
        resEntityUnits = 'ph/s/.1%bw'

    resLabelsToSave = ['Photon Energy', 'Horizontal Position', 'Vertical Position', resEntityName] #OC26042016
    resUnitsToSave=['eV', 'm', 'm', resEntityUnits] #OC26042016

    if(((rank == 0) or (nProc == 1)) and (_opt_bl != None)): #calculate once the central wavefront in the master process (this has to be done only if propagation is required)

        if(useGsnBmSrc):
            srwl.CalcElecFieldGaussian(wfr, _mag, arPrecParSR)
            #print('DEBUG: Commented-out: CalcElecFieldGaussian')
        else:

            #print('Single-electron SR calculation ... ', end='') #DEBUG
            #t0 = time.time(); #DEBUG

            srwl.CalcElecFieldSR(wfr, 0, _mag, arPrecParSR)

            #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG
            #print('DEBUG MESSAGE: CalcElecFieldSR called (rank:', rank,')')

        #print('DEBUG MESSAGE: Central Wavefront calculated')

        #print('Wavefront propagation calculation ... ', end='') #DEBUG
        #t0 = time.time(); #DEBUG

        #if(_w_wr != 0.): #OC26032016
        if(_wr != 0.): #OC07092016
            wfr.Rx = _wr
            wfr.Ry = _wr

        srwl.PropagElecField(wfr, _opt_bl)

        #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG
        #meshRes.set_from_other(wfr.mesh) #DEBUG
        #resStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny, doMutual) #DEBUG
        #wfr.calc_stokes(resStokes) #DEBUG
        #srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1, _mutual = doMutual) #DEBUG

        #print('DEBUG: Commented-out: PropagElecField')
        #print('DEBUG MESSAGE: Central Wavefront propagated')
        if(_pres_ang > 0):
            srwl.SetRepresElecField(wfr, 'a')
            #print('DEBUG: Commented-out: SetRepresElecField')

        meshRes.set_from_other(wfr.mesh)

        if(doMutual > 0):
            if(_char == 2):
                meshRes.ny = 1
                meshRes.yStart = _y0
                meshRes.yFin = _y0
            elif(_char == 3):
                meshRes.nx = 1
                meshRes.xStart = _x0
                meshRes.xFin = _x0

        if(nProc > 1): #send resulting mesh to all workers
            #comMPI.send(wfr.mesh, dest=)
            arMesh = array('f', [meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny])
            #comMPI.Bcast([arMesh, MPI.FLOAT], root=MPI.ROOT)
            #comMPI.Bcast([arMesh, MPI.FLOAT])

            #print('DEBUG MESSAGE: Rank0 is about to broadcast mesh of Propagated central wavefront')
            for iRank in range(nProc - 1):
                dst = iRank + 1
                #print("msg %d: sending data from %d to %d" % (iRank, rank, dst)) #an he
                comMPI.Send([arMesh, MPI.FLOAT], dest=dst)
            #print('DEBUG MESSAGE: Mesh of Propagated central wavefront broadcasted')

        #DEBUG
        #print('meshRes: ne=', meshRes.ne, 'eStart=', meshRes.eStart, 'eFin=', meshRes.eFin)
        #END DEBUG

        resStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny, doMutual)
        #wfr.calc_stokes(resStokes) #OC190414 (don't take into account first "central" beam)
        workStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny, doMutual)

        #iAvgProc += 1 #OC190414 (commented-out)
        #iSave += 1

    #slaves = [] #an he
    #print('DEBUG MESSAGE: rank=', rank)
    if((rank > 0) or (nProc == 1)):

        if((nProc > 1) and (_opt_bl != None)): #receive mesh for the resulting wavefront from the master
            arMesh = array('f', [0]*9)
            #_stat = MPI.Status() #an he
            #comMPI.Recv([arMesh, MPI.FLOAT], source=0)
            comMPI.Recv([arMesh, MPI.FLOAT], source=MPI.ANY_SOURCE)
            #comMPI.Bcast([arMesh, MPI.FLOAT], root=0)
            #print("received mesh %d -> %d" % (_stat.Get_source(), rank))
            meshRes.eStart = arMesh[0]
            meshRes.eFin = arMesh[1]
            meshRes.ne = int(arMesh[2])
            meshRes.xStart = arMesh[3]
            meshRes.xFin = arMesh[4]
            meshRes.nx = int(arMesh[5])
            meshRes.yStart = arMesh[6]
            meshRes.yFin = arMesh[7]
            meshRes.ny = int(arMesh[8])
            #sys.exit(0)

        nRadPt = meshRes.ne*meshRes.nx*meshRes.ny
        if(doMutual > 0): nRadPt *= nRadPt

        nStPt = nRadPt*4
        randAr = array('d', [0]*6) #for random Gaussian numbers

        #random.seed(rank)
        random.seed(rank*123)
        newSeed = random.randint(0, 1000000)
        random.seed(newSeed)

        iAuxSendCount = 0 #for debug

        for i in range(nPartPerProc): #loop over macro-electrons

            if(_rand_meth == 1):
                for ir in range(5): #to expend to 6D eventually
                    randAr[ir] = random.gauss(0, 1)
            elif(_rand_meth == 2):
                if(nProc > 1):
                    iArg = i*(nProc - 1) + rank
                    a1 = srwl_uti_math_seq_halton(iArg, 2)
                    a2 = srwl_uti_math_seq_halton(iArg, 3)
                    a3 = srwl_uti_math_seq_halton(iArg, 5)
                    a4 = srwl_uti_math_seq_halton(iArg, 7)
                    a5 = srwl_uti_math_seq_halton(iArg, 11) #?
                elif(nProc == 1):
                    i_p_1 = i + 1
                    a1 = srwl_uti_math_seq_halton(i_p_1, 2)
                    a2 = srwl_uti_math_seq_halton(i_p_1, 3)
                    a3 = srwl_uti_math_seq_halton(i_p_1, 5)
                    a4 = srwl_uti_math_seq_halton(i_p_1, 7)
                    a5 = srwl_uti_math_seq_halton(i_p_1, 11) #?
                twoPi = 2*pi
                twoPi_a2 = twoPi*a2
                twoPi_a4 = twoPi*a4
                m2_log_a1 = -2.0*log(a1)
                m2_log_a3 = -2.0*log(a3)
                randAr[0] = sqrt(m2_log_a1)*cos(twoPi_a2)
                randAr[1] = sqrt(m2_log_a1)*sin(twoPi_a2)
                randAr[2] = sqrt(m2_log_a3)*cos(twoPi_a4)
                randAr[3] = sqrt(m2_log_a3)*sin(twoPi_a4)
                randAr[4] = sqrt(m2_log_a1)*cos(twoPi*a3) #or just random.gauss(0,1) depends on cases #why not using a5?
                randAr[5] = a5
            elif(_rand_meth == 3):
                #to program LPtau sequences here
                continue

            #DEBUG
            #if(i == 0):
            #    randAr = array('d', [0,0,0,2,0])
            #if(i == 1):
            #    randAr = array('d', [0,0,0,-2,0])
            #END DEBUG

            auxPXp = SigQX*randAr[0]
            auxPX = SigPX*randAr[1] + AX*auxPXp/GX
            wfr.partBeam.partStatMom1.x = elecX0 + auxPX
            wfr.partBeam.partStatMom1.xp = elecXp0 + auxPXp
            auxPYp = SigQY*randAr[2]
            auxPY = SigPY*randAr[3] + AY*auxPYp/GY
            wfr.partBeam.partStatMom1.y = elecY0 + auxPY
            wfr.partBeam.partStatMom1.yp = elecYp0 + auxPYp
            #wfr.partBeam.partStatMom1.gamma = (elecEn0 + elecAbsEnSpr*randAr[4])/0.51099890221e-03 #Relative Energy
            wfr.partBeam.partStatMom1.gamma = elecGamma0*(1 + elecAbsEnSpr*randAr[4]/elecE0)

            #reset mesh, because it may be modified by CalcElecFieldSR and PropagElecField
            #print('Numbers of points (before re-setting): nx=', wfr.mesh.nx, ' ny=', wfr.mesh.ny) #DEBUG
            curWfrMesh = wfr.mesh #OC02042016
            if((curWfrMesh.ne != _mesh.ne) or (curWfrMesh.nx != _mesh.nx) or (curWfrMesh.ny != _mesh.ny)):
                wfr.allocate(_mesh.ne, _mesh.nx, _mesh.ny)

            wfr.mesh.set_from_other(_mesh)
            #print('Numbers of points (after re-setting): nx=', wfr.mesh.nx, ' ny=', wfr.mesh.ny) #DEBUG

            if(_e_ph_integ == 1):
                if(_rand_meth == 1):
                    ePh = random.uniform(_mesh.eStart, _mesh.eFin)
                else:
                    ePh = _mesh.eStart + (_mesh.eFin - _mesh.eStart)*randAr[5]

                wfr.mesh.eStart = ePh
                wfr.mesh.eFin = ePh
                wfr.mesh.ne = 1

            wfr.presCA = 0 #presentation/domain: 0- coordinates, 1- angles
            wfr.presFT = 0 #presentation/domain: 0- frequency (photon energy), 1- time

            if(nProc == 1):
                print('i=', i, 'Electron Coord.: x=', wfr.partBeam.partStatMom1.x, 'x\'=', wfr.partBeam.partStatMom1.xp, 'y=', wfr.partBeam.partStatMom1.y, 'y\'=', wfr.partBeam.partStatMom1.yp, 'E=',  wfr.partBeam.partStatMom1.gamma*0.51099890221e-03)
                if(_e_ph_integ == 1):
                     print('Eph=', wfr.mesh.eStart)

                phase_space_electrons[0,i] = wfr.partBeam.partStatMom1.x
                phase_space_electrons[1,i] = wfr.partBeam.partStatMom1.xp
                phase_space_electrons[2,i] = wfr.partBeam.partStatMom1.y
                phase_space_electrons[3,i] = wfr.partBeam.partStatMom1.yp
                phase_space_electrons[4,i] = wfr.partBeam.partStatMom1.gamma*0.51099890221e-03
                phase_space_electrons[5,i] = wfr.mesh.eStart


            if(calcSpecFluxSrc): #consider taking into account _rand_meth != 1 here
                xObs = random.uniform(_mesh.xStart, _mesh.xFin)
                wfr.mesh.xStart = xObs
                wfr.mesh.xFin = xObs
                yObs = random.uniform(_mesh.yStart, _mesh.yFin)
                wfr.mesh.yStart = yObs
                wfr.mesh.yFin = yObs
                #print('xObs=', xObs, 'yObs=', yObs)

            try:
                if(useGsnBmSrc):
                    _mag.x = wfr.partBeam.partStatMom1.x
                    _mag.xp = wfr.partBeam.partStatMom1.xp
                    _mag.y = wfr.partBeam.partStatMom1.y
                    _mag.yp = wfr.partBeam.partStatMom1.yp
                    srwl.CalcElecFieldGaussian(wfr, _mag, arPrecParSR)
                    #print('DEBUG: Commented-out: CalcElecFieldGaussian')
                    #print('Gaussian wavefront calc. done')
                else:

                    #print('Single-electron SR calculatiton ... ', end='') #DEBUG
                    #t0 = time.time(); #DEBUG
                    #print('Numbers of points: nx=', wfr.mesh.nx, ' ny=', wfr.mesh.ny) #DEBUG



                    srwl.CalcElecFieldSR(wfr, 0, _mag, arPrecParSR) #calculate Electric Field emitted by current electron

                    if save_individual_electrons:
                        if filename is not None:
                            dump_wfr(fn,wfr,"source%d"%i)

                    #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG
                    #print('DEBUG: Commented-out: CalcElecFieldSR')
                    #print('DEBUG MESSAGE: CalcElecFieldSR called (rank:', rank,')')

                if(_opt_bl != None):

                    #print('Wavefront propagation calculation ... ', end='') #DEBUG
                    #t0 = time.time(); #DEBUG

                    #if(_w_wr != 0.): #OC26032016
                    if(_wr != 0.): #OC07092016
                        wfr.Rx = _wr
                        wfr.Ry = _wr


                    srwl.PropagElecField(wfr, _opt_bl) #propagate Electric Field emitted by the electron
                    if save_individual_electrons:
                        if filename is not None:
                            dump_wfr(fn,wfr,"beamline%d"%i)

                            fn["limits%d"%i] = numpy.array([wfr.mesh.xStart,wfr.mesh.xFin,wfr.mesh.nx,
                                                            wfr.mesh.yStart,wfr.mesh.yFin,wfr.mesh.ny,])
                    #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG
                    #print('DEBUG: Commented-out: PropagElecField')

                if(_pres_ang > 0):
                    srwl.SetRepresElecField(wfr, 'a')
                    #print('DEBUG: Commented-out: SetRepresElecField')

            except:
                traceback.print_exc()

            meshWork = deepcopy(wfr.mesh)

            if(doMutual > 0):
                if(_char == 2):
                    meshWork.ny = 1
                    meshWork.yStart = _y0
                    meshWork.yFin = _y0
                elif(_char == 3):
                    meshWork.nx = 1
                    meshWork.xStart = _x0
                    meshWork.xFin = _x0

            if(workStokes == None):
                workStokes = SRWLStokes(1, 'f', meshWork.eStart, meshWork.eFin, meshWork.ne, meshWork.xStart, meshWork.xFin, meshWork.nx, meshWork.yStart, meshWork.yFin, meshWork.ny, doMutual)
            else:
                nRadPtCur = meshWork.ne*meshWork.nx*meshWork.ny
                if(doMutual > 0):
                    nRadPtCur *= nRadPtCur

                nPtCur = nRadPtCur*4

                if(len(workStokes.arS) < nPtCur):
                    del workStokes.arS
                    workStokes.arS = array('f', [0]*nPtCur)
                    #workStokes.mesh.set_from_other(wfr.mesh)

            print(">>>> calculating Stokes for electron ",i)
            wfr.calc_stokes(workStokes) #calculate Stokes parameters from Electric Field


            #DEBUG
            #srwl_uti_save_intens_ascii(workStokes.arS, workStokes.mesh, _file_path, 1)
            #END DEBUG

            if(resStokes == None):
                resStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny, doMutual)
                #DEBUG
                #print('resStokes #2: ne=', resStokes.mesh.ne, 'eStart=', resStokes.mesh.eStart, 'eFin=', resStokes.mesh.eFin)
                #END DEBUG

            if(_opt_bl == None):
                #resStokes.avg_update_same_mesh(workStokes, iAvgProc, 1)

                #print('resStokes.avg_update_same_mesh ... ', end='') #DEBUG
                #t0 = time.time(); #DEBUG

                resStokes.avg_update_same_mesh(workStokes, iAvgProc, 1, ePhIntegMult) #to treat all Stokes components / Polarization in the future

                #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG
                #DEBUG
                #srwl_uti_save_intens_ascii(workStokes.arS, workStokes.mesh, _file_path, 1)
                #END DEBUG

            else:
                #print('DEBUG MESSAGE: Started interpolation of current wavefront on resulting mesh')
                #if(doMutual <= 0): resStokes.avg_update_interp(workStokes, iAvgProc, 1, 1)
                #else: resStokes.avg_update_interp_mutual(workStokes, iAvgProc, 1)

                #print('resStokes.avg_update_interp ... ', end='') #DEBUG
                #t0 = time.time(); #DEBUG

                if(doMutual <= 0): resStokes.avg_update_interp(workStokes, iAvgProc, 1, 1, ePhIntegMult) #to treat all Stokes components / Polarization in the future
                else: resStokes.avg_update_interp_mutual(workStokes, iAvgProc, 1, ePhIntegMult)

                #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG
                #print('DEBUG MESSAGE: Finished interpolation of current wavefront on resulting mesh')

            iAvgProc += 1
            if(iAvgProc >= _n_part_avg_proc):
                if(nProc > 1):
                    #sys.exit(0)
                    #print("sending data from %d to 0" % rank) #an he
                    #DEBUG
                    #srwl_uti_save_intens_ascii(resStokes.arS, resStokes.mesh, _file_path, 1)
                    #END DEBUG

                    #DEBUG
                    #srwl_uti_save_text("Preparing to sending # " + str(iAuxSendCount + 1), _file_path + "." + str(rank) + "bs.dbg")
                    #END DEBUG

                    comMPI.Send([resStokes.arS, MPI.FLOAT], dest=0)
                    iAuxSendCount += 1 #for debug

                    #DEBUG
                    #srwl_uti_save_text("Sent # " + str(iAuxSendCount), _file_path + "." + str(rank) + "es.dbg")
                    #END DEBUG

                    for ir in range(nStPt):
                        resStokes.arS[ir] = 0
                    #DEBUG
                    #srwl_uti_save_intens_ascii(resStokes.arS, resStokes.mesh, _file_path, 1)
                    #END DEBUG

                iAvgProc = 0

            if(nProc == 1):
                #DEBUG
                #if(i == 1):
                #    srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1)
                #    sys.exit(0)
                #END DEBUG
                iSave += 1
                if((_file_path != None) and (iSave == _n_save_per)):
                    #Saving results from time to time in the process of calculation:

                    #print('srwl_uti_save_intens_ascii ... ', end='') #DEBUG
                    #t0 = time.time(); #DEBUG

                    #srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1, _mutual = doMutual)
                    srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1, _arLabels = resLabelsToSave, _arUnits = resUnitsToSave, _mutual = doMutual) #OC26042016

                    #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG

                    #sys.exit(0)
                    iSave = 0

    elif((rank == 0) and (nProc > 1)):

        #nRecv = int(nPartPerProc*nProc/_n_part_avg_proc + 1e-09)
        nRecv = nSentPerProc*(nProc - 1) #Total number of sending acts to be made by all worker processes, and to be received by master

        print('DEBUG MESSAGE: Actual number of macro-electrons:', nRecv*_n_part_avg_proc)

        #DEBUG
        #srwl_uti_save_text("nRecv: " + str(nRecv) + " nPartPerProc: " + str(nPartPerProc) + " nProc: " + str(nProc) + " _n_part_avg_proc: " + str(_n_part_avg_proc), _file_path + ".00.dbg")
        #END DEBUG

        if(resStokes == None):
            resStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny, doMutual)

        if(workStokes == None):
            workStokes = SRWLStokes(1, 'f', meshRes.eStart, meshRes.eFin, meshRes.ne, meshRes.xStart, meshRes.xFin, meshRes.nx, meshRes.yStart, meshRes.yFin, meshRes.ny, doMutual)

        #MR29092016 #Erase the contents of .log file:
        #OC: Consider implementing this
        total_num_of_particles = nRecv * _n_part_avg_proc
        #srwl_save_status(0, total_num_of_particles, cores=nProc, particles_per_iteration=_n_part_avg_proc)
        srwl_uti_save_stat_wfr_emit_prop_multi_e(0, total_num_of_particles, cores=nProc, particles_per_iteration=_n_part_avg_proc)

        for i in range(nRecv): #loop over messages from workers

            #DEBUG
            #srwl_uti_save_text("Preparing to receiving # " + str(i), _file_path + ".br.dbg")
            #END DEBUG

            comMPI.Recv([workStokes.arS, MPI.FLOAT], source=MPI.ANY_SOURCE) #receive #an he (commented-out)

            #MR20160907 #Save .log and .json files:
            particle_number = (i + 1) * _n_part_avg_proc
            #srwl_save_status(particle_number, total_num_of_particles)
            srwl_uti_save_stat_wfr_emit_prop_multi_e(particle_number, total_num_of_particles)

            #DEBUG
            #srwl_uti_save_text("Received intensity # " + str(i), _file_path + ".er.dbg")
            #END DEBUG

            #resStokes.avg_update_same_mesh(workStokes, i + 1)
            #resStokes.avg_update_same_mesh(workStokes, i + 1, 1, ePhIntegMult) #to treat all Stokes components / Polarization in the future
            multFinAvg = 1 if(_n_part_avg_proc > 1) else ePhIntegMult #OC120714 fixed: the normalization may have been already applied at the previous avaraging on each worker node!

            #print('resStokes.avg_update_same_mesh ... ', end='') #DEBUG
            #t0 = time.time(); #DEBUG

            resStokes.avg_update_same_mesh(workStokes, i + 1, 1, multFinAvg) #in the future treat all Stokes components / Polarization, not just s0!

            #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG
            #DEBUG
            #srwl_uti_save_text("Updated Stokes after receiving intensity # " + str(i), _file_path + "." + str(i) + "er.dbg")
            #END DEBUG

            iSave += 1
            if(iSave == _n_save_per):
                #Saving results from time to time in the process of calculation

                #print('srwl_uti_save_intens_ascii ... ', end='') #DEBUG
                #t0 = time.time(); #DEBUG

                #srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1, _mutual = doMutual)
                srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1, _arLabels = resLabelsToSave, _arUnits = resUnitsToSave, _mutual = doMutual) #OC26042016

                #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG

                iSave = 0

    #DEBUG
    #srwl_uti_save_text("Exiting srwl_wfr_emit_prop_multi_e", _file_path + "." + str(rank) + "e.dbg")
    #END DEBUG


    if((rank == 0) or (nProc == 1)):
        if filename is not None:
            arxx = numpy.array(resStokes.arS)
            arxx = arxx.reshape((4,resStokes.mesh.ny,resStokes.mesh.nx)).T
            fn["limits_multielectron_stokes"] = [resStokes.mesh.xStart,resStokes.mesh.xFin,resStokes.mesh.nx,
                                            resStokes.mesh.yStart,resStokes.mesh.yFin,resStokes.mesh.ny]
            fn["multielectron_stokes0"] = arxx[:,:,0]
            fn["multielectron_stokes1"] = arxx[:,:,1]
            fn["multielectron_stokes2"] = arxx[:,:,2]
            fn["multielectron_stokes3"] = arxx[:,:,3]

            fn["phase_space_electrons"] = phase_space_electrons

            dump_close(fn)

    if((rank == 0) or (nProc == 1)):
        #Saving final results:
        if(_file_path != None):

            #print('srwl_uti_save_intens_ascii ... ', end='') #DEBUG
            #t0 = time.time(); #DEBUG

            #srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1, _mutual = doMutual)
            srwl_uti_save_intens_ascii(resStokes.arS, meshRes, _file_path, 1, _arLabels = resLabelsToSave, _arUnits = resUnitsToSave, _mutual = doMutual) #OC26042016

            #print('completed (lasted', round(time.time() - t0, 6), 's)') #DEBUG

        return resStokes
    else:
        return None