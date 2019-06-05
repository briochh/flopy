# Test multi-species options in mt3d
import os
import numpy as np
import flopy

testpth = os.path.join('.', 'temp', 't023')
# make the directory if it does not exist
if not os.path.isdir(testpth):
    os.makedirs(testpth)

def test_mt3d_multispecies():
    # modflow model
    modelname = 'multispecies'
    nlay = 1
    nrow = 20
    ncol = 20
    nper = 10
    mf = flopy.modflow.Modflow(modelname=modelname, model_ws=testpth)
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol,
                                   nper=nper)
    lpf = flopy.modflow.ModflowLpf(mf)
    rch = flopy.modflow.ModflowRch(mf)
    evt = flopy.modflow.ModflowEvt(mf)
    mf.write_input()

    # Create a 5-component mt3d model and write the files
    ncomp = 5
    mt = flopy.mt3d.Mt3dms(modelname=modelname, modflowmodel=mf,
                           model_ws=testpth, verbose=True)
    sconc3 = np.random.random((nrow, ncol))
    btn = flopy.mt3d.Mt3dBtn(mt, ncomp=ncomp, sconc=1., sconc2=2.,
                             sconc3=sconc3, sconc5=5.)
    crch32 = np.random.random((nrow, ncol))
    cevt33 = np.random.random((nrow, ncol))
    
    point_data = [
        [0, 9, 9] + [np.multiply(np.random.random(5), 1e-3).tolist()],
        [0, 7, 7] + [np.random.random(5).tolist()]]
    sp_data = []
    for k, i, j, d in point_data:
        sp_data.append([k, i, j, d[0], 15] + [r for r in d])

    sp_data = np.rec.fromarrays(
        np.array(sp_data).T,
        dtype=np.dtype([('k', np.int32), ('i', np.int32),
                        ('j', np.int32), ('css', np.float32),
                        ('itype', np.int32),
                        ('cssm(01)', np.float32), ('cssm(02)', np.float32),
                        ('cssm(03)', np.float32), ('cssm(04)', np.float32),
                        ('cssm(05)', np.float32)]))
    sp_data_full = {i: sp_data for i in range(mt.nper)}
    ssm = flopy.mt3d.Mt3dSsm(mt, crch=1., crch2=2., crch3={2:crch32}, crch5=5.,
                             cevt=1., cevt2=2., cevt3={3:cevt33}, cevt5=5.,
                             stress_period_data=sp_data_full)
    crch2 = ssm.crch[1].array
    assert(crch2.max() == 2.)
    cevt2 = ssm.cevt[1].array
    assert(cevt2.max() == 2.)
    mt.write_input()

    mt_load = flopy.mt3d.Mt3dms.load('{}.nam'.format(modelname),
                                     modflowmodel=mf,
                                     model_ws=testpth, verbose=True,
                                     load_only=['btn'])
    ext_unit_dict = flopy.utils.mfreadnam.parsenamefile(
        os.path.join(mt.model_ws, mt.namefile), packages='SSM')
    ssm_loaded = flopy.mt3d.Mt3dSsm.load(
        os.path.join(mt.model_ws, mt.ssm.file_name[0]), model=mt_load,
        ext_unit_dict=ext_unit_dict)
    o_sp = mt.ssm.stress_period_data.data
    n_sp = mt_load.ssm.stress_period_data.data
    assert np.all([np.isclose(o_sp[kper][name], n_sp[kper][name]).all()
                   for kper in range(mt.nper)
                   for name in o_sp[0].dtype.fields.keys()])
    # Create a second MODFLOW model
    modelname2 = 'multispecies2'
    mf2 = flopy.modflow.Modflow(modelname=modelname2, model_ws=testpth)
    dis2 = flopy.modflow.ModflowDis(mf2, nlay=nlay, nrow=nrow, ncol=ncol,
                                    nper=nper)

    # Load the MT3D model into mt2 and then write it out
    fname = modelname + '.nam'
    mt2 = flopy.mt3d.Mt3dms.load(fname, model_ws=testpth, verbose=True)
    mt2.name = modelname2
    mt2.write_input()

    return


if __name__ == '__main__':
    test_mt3d_multispecies()
