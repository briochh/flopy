import sys
from flopy.mbase import Package

class Mt3dGcg(Package):
    """
    MT3DMS Generalized Conjugate Gradient Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to which
        this package will be added.
    mxiter : int
        is the maximum number of outer iterations; it should be set to an
        integer greater than one only when a nonlinear sorption isotherm is
        included in simulation. (default is 1)
    iter1 : int
        is the maximum number of inner iterations; a value of 30-50 should be
        adequate for most problems. (default is 50)
    isolve : int
        is the type of preconditioners to be used with the Lanczos/ORTHOMIN
        acceleration scheme:
        = 1, Jacobi
        = 2, SSOR
        = 3, Modified Incomplete Cholesky (MIC) (MIC usually converges faster,
        but it needs significantly more memory)
        (default is 3)
    ncrs : int
        is an integer flag for treatment of dispersion tensor cross terms:
        = 0, lump all dispersion cross terms to the right-hand-side
        (approximate but highly efficient). = 1, include full dispersion
        tensor (memory intensive).
        (default is 0)
    accl : float
        is the relaxation factor for the SSOR option; a value of 1.0 is
        generally adequate.
        (default is 1)
    cclose : float
        is the convergence criterion in terms of relative concentration; a
        real value between 10-4 and 10-6 is generally adequate.
        (default is 1.E-5)
    iprgcg : int
        IPRGCG is the interval for printing the maximum concentration changes
        of each iteration. Set IPRGCG to zero as default for printing at the
        end of each stress period.
        (default is 0)
    extension : string
        Filename extension (default is 'gcg')
    unitnumber : int
        File unit number (default is 35).

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.mt3d.Mt3dms()
    >>> gcg = flopy.mt3d.Mt3dGcg(m)

    """
    def __init__(self, model, mxiter=1, iter1=50, isolve=3, ncrs=0,
                 accl=1, cclose=1e-5, iprgcg=0, extension='gcg',
                 unitnumber=35):
        Package.__init__(self, model, extension, 'GCG', unitnumber)
        self.mxiter = mxiter
        self.iter1 = iter1
        self.isolve = isolve
        self.ncrs = ncrs
        self.accl = accl
        self.cclose = cclose
        self.iprgcg = iprgcg
        self.parent.add_package(self)
        return
        
    def write_file(self):
        # Open file for writing
        f_gcg = open(self.fn_path, 'w')
        f_gcg.write('{} {} {} {}\n' %
                    (self.mxiter, self.iter1, self.isolve, self.ncrs))
        f_gcg.write('{} {} {}\n' %
                   (self.accl, self.cclose, self.iprgcg))
        f_gcg.close()
        return

    @staticmethod
    def load(f, model, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        gcg :  Mt3dGcg object
            Mt3dGcg object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> gcg = flopy.mt3d.Mt3dGcg.load('test.gcg', m)

        """

        if model.verbose:
            sys.stdout.write('loading gcg package file...\n')

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Dataset 0 -- comment line
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # Item F1: MIXELM, PERCEL, MXPART, NADVFD - line already read above
        if model.verbose:
            print('   loading MXITER, ITER1, ISOLVE, NCRS...')
        t = line.strip().split()
        mxiter = int(t[0])
        iter1 = int(t[1])
        isolve = int(t[2])
        ncrs = int(t[3])
        if model.verbose:
            print('   MXITER {}'.format(mxiter))
            print('   ITER1 {}'.format(iter1))
            print('   ISOLVE {}'.format(isolve))
            print('   NCRS {}'.format(ncrs))

        # Item F2: ACCL, CCLOSE, IPRGCG
        if model.verbose:
            print('   loading ACCL, CCLOSE, IPRGCG...')
        line = f.readline()
        t = line.strip().split()
        accl = float(t[0])
        cclose = float(t[1])
        iprgcg = int(t[2])
        if model.verbose:
            print('   ACCL {}'.format(accl))
            print('   CCLOSE {}'.format(cclose))
            print('   IPRGCG {}'.format(iprgcg))

        # Construct and return gcg package
        gcg = Mt3dGcg(model, mxiter=mxiter, iter1=iter1, isolve=isolve,
                      ncrs=ncrs, accl=accl, cclose=cclose, iprgcg=iprgcg)
        return gcg
