import numpy as np
import struct
import string
import glob,os,re
import astropy.constants as ac
import astropy.units as au

import cooling
coolftn=cooling.coolftn()

def parse_filename(filename):
    """
    #   PARSE_FILENAME    Break up a full-path filename into its component
    #   parts to check the extension, make it more readable, and extract the step
    #   number.
    #
    #   PATH,BASENAME,STEP,EXT = PARSE_FILENAME(FILENAME)
    #
    #   E.g. If FILENAME='/home/Blast.0000.bin', then PATH='/home',
    #   BASENAME='Blast', STEP='0000', and EXT='bin'.
    #
    """


    path=os.path.dirname(filename)
    if path[-3:] == 'id0':
        path=path[:-3]
        mpi_mode=True
    else:
        path=path+os.path.sep
        mpi_mode=False

    base=os.path.basename(filename)
    base_split=base.split('.')
    if len(base_split) == 3:
        id=base_split[0]
        step=base_split[1]
        ext=base_split[2]
    else:
        id='.'.join(base_split[:-2])
        step=base_split[-2]
        ext=base_split[-1]

    return path,id,step,ext,mpi_mode

def parse_line(line, grid):
    sp = line.strip().split()

    if b"vtk" in sp:
        grid['vtk_version'] = sp[-1]
    elif b"time=" in sp:
        time_index = sp.index(b"time=")
        grid['time'] = float(sp[time_index+1].rstrip(b','))
        if b'level' in sp: grid['level'] = int(sp[time_index+3].rstrip(b','))
        if b'domain' in sp: grid['domain'] = int(sp[time_index+5].rstrip(b','))
        if sp[0] == b"PRIMITIVE":
            grid['prim_var_type']=True
    elif b"DIMENSIONS" in sp:
        grid['Nx'] = np.array(sp[-3:]).astype('int')
    elif b"ORIGIN" in sp:
        grid['left_edge'] = np.array(sp[-3:]).astype('float64')
    elif b"SPACING" in sp:
        grid['dx'] = np.array(sp[-3:]).astype('float64')
    elif b"CELL_DATA" in sp:
        grid['ncells'] = int(sp[-1])
    elif b"SCALARS" in sp:
        grid['read_field'] = sp[1]
        grid['read_type'] = 'scalar'
    elif b"VECTORS" in sp:
        grid['read_field'] = sp[1]
        grid['read_type'] = 'vector'
    elif b"NSTARS" in sp:
        grid['nstar'] = eval(sp[1])
    elif b"POINTS" in sp:
        grid['nstar'] = eval(sp[1])
        grid['ncells'] = eval(sp[1])


class AthenaDomain(object):
    def __init__(self,filename,ds=None,setgrid=True,serial=False):
        self.flist = glob.glob(filename)
        if len(self.flist) == 0:
            print(('no such file: %s' % filename))
        dir, id, step, ext, mpi = parse_filename(filename)
        self.dir = dir
        self.id = id
        self.step = step
        self.ext = ext
        self.starfile = os.path.join(dir+'id0/','%s.%s.%s.%s' % (id,step,'starpar',ext))
        if serial: mpi = False
        self.mpi = mpi
        self.ngrids = 1
        if mpi:
            self.ngrids += len(glob.glob(os.path.join(dir,'id*/%s-id*.%s.%s' % (id, step, ext))))
            for n in range(1,self.ngrids):
                self.flist.append(os.path.join(dir,'id%d/%s-id%d.%s.%s' % (n,id,n,step, ext)))
        if setgrid:
            if ds==None:
                self.grids=self._setup_grid()
            else:
                if ds.grids[0]['filename'] != self.flist[0]:
                    for g,f in zip(ds.grids,self.flist): g['filename']=f
                self.grids=ds.grids
            self.domain=self._setup_domain(self.grids)
            if ds==None:
                self.domain['field_map']=None
            else:
                self.domain['field_map']=ds.domain['field_map']
            self._setup_mpi_grid()
            self._setup()

    def _setup(self):
        self.domain['data']={}

    def _setup_domain(self,grids):
        domain = {}
        ngrids=len(grids)
        left_edges = np.empty((ngrids,3), dtype='float32')
        dxs = np.empty((ngrids,3), dtype='float32')
        Nxs = np.ones_like(dxs)
        for nproc,g in enumerate(grids):
            left_edges[nproc,:] = g['left_edge']
            Nxs[nproc,:] = g['Nx']
            dxs[nproc,:] = g['dx']

        right_edges = left_edges + Nxs*dxs

        left_edge = left_edges.min(0)
        right_edge = right_edges.max(0)

        domain['left_edge'] = left_edge
        domain['right_edge'] = right_edge
        domain['dx'] = dxs[0,:]
        domain['Lx'] = right_edge - left_edge
        domain['center'] = 0.5*(right_edge + left_edge)
        domain['Nx'] = np.round(domain['Lx']/domain['dx']).astype('int')
        domain['ndim'] = 3 # should be revised
        file = open(self.flist[0],'rb')
        tmpgrid = {}
        tmpgrid['time']=None
        while tmpgrid['time'] is None:
            line = file.readline()
            parse_line(line,tmpgrid)
        file.close()
        domain['time'] = tmpgrid['time']

        return domain

    def _setup_mpi_grid(self):
        gnx = self.grids[0]['Nx']
        self.NGrids = (self.domain['Nx']/self.grids[0]['Nx']).astype(np.int)
        self.gid = np.arange(self.ngrids)
        i = 0
        for n in range(self.NGrids[2]):
            for m in range(self.NGrids[1]):
                for l in range(self.NGrids[0]):
                    self.grids[i]['is']=np.array([l*gnx[0],m*gnx[1],n*gnx[2]])
                    i += 1

    def _setup_grid(self):
        grids=[]
        for nproc in range(self.ngrids):
            file = open(self.flist[nproc],'rb')
            grid = {}
            grid['filename']=self.flist[nproc]
            grid['read_field'] = None
            grid['read_type'] = None
            while grid['read_field'] is None:
                grid['data_offset']=file.tell()
                line = file.readline()
                parse_line(line, grid)
            file.close()
            grid['Nx'] -= 1
            grid['Nx'][grid['Nx'] == 0] = 1
            grid['dx'][grid['Nx'] == 1] = 1.
            grid['right_edge'] = grid['left_edge'] + grid['Nx']*grid['dx']
            #grid['field_map']=None

            grids.append(grid)

        return grids

class AthenaDataSet(AthenaDomain):
    def _setup(self):
        for i,g in enumerate(self.grids):
            g['data']={}
        if self.domain['field_map']==None:
            self.domain['field_map'] = self._set_field_map(self.grids[0])
        fm = self.domain['field_map']
        if 'cell_centered_B' in list(fm.keys()):
            fm['magnetic_field']=fm['cell_centered_B']
            fm.pop('cell_centered_B')
        elif 'face_centered_B' in list(fm.keys()):
            fm['magnetic_field']=fm['face_centered_B']
            fm.pop('face_centered_B')
        nscal=0
        if 'specific_scalar[0]' in list(fm.keys()):
            keys=list(fm.keys())
            for k in keys:
                if k.startswith('specific_scalar'):
                    newkey=re.sub("\s|\W","",k)
                    fm[newkey] = fm.pop(k)
                    nscal += 1
        self.field_list=list(fm.keys())

        derived_field_list=[]
        derived_field_list_hd=[]
        derived_field_list_mhd=[]
        if 'magnetic_field' in self.field_list:
            derived_field_list_mhd.append('magnetic_field1')
            derived_field_list_mhd.append('magnetic_field2')
            derived_field_list_mhd.append('magnetic_field3')
            derived_field_list_mhd.append('magnetic_energy1')
            derived_field_list_mhd.append('magnetic_energy2')
            derived_field_list_mhd.append('magnetic_energy3')
            derived_field_list_mhd.append('magnetic_pressure')
            derived_field_list_mhd.append('plasma_beta')
            derived_field_list_mhd.append('alfven_velocity1')
            derived_field_list_mhd.append('alfven_velocity2')
            derived_field_list_mhd.append('alfven_velocity3')
            derived_field_list_mhd.append('magnetic_stress')
            derived_field_list_mhd.append('magnetic_stress1')
            derived_field_list_mhd.append('magnetic_stress2')
            derived_field_list_mhd.append('magnetic_stress3')
        if 'velocity' in self.field_list:
            derived_field_list_hd.append('velocity1')
            derived_field_list_hd.append('velocity2')
            derived_field_list_hd.append('velocity3')
            derived_field_list_hd.append('velocity_magnitude')
            derived_field_list_hd.append('kinetic_energy1')
            derived_field_list_hd.append('kinetic_energy2')
            derived_field_list_hd.append('kinetic_energy3')
            derived_field_list_hd.append('momentum1')
            derived_field_list_hd.append('momentum2')
            derived_field_list_hd.append('momentum3')
            derived_field_list_hd.append('reynold_stress')
            derived_field_list_hd.append('reynold_stress1')
            derived_field_list_hd.append('reynold_stress2')
            derived_field_list_hd.append('reynold_stress3')
        if 'pressure' in self.field_list:
            derived_field_list_hd.append('sound_speed')
            derived_field_list_hd.append('temperature')
            derived_field_list_hd.append('T1')
        if 'gravitational_potential' in self.field_list:
            derived_field_list_hd.append('potential_energy')
            derived_field_list_hd.append('gravity_stress')
            derived_field_list_hd.append('gravity_stress1')
            derived_field_list_hd.append('gravity_stress2')
            derived_field_list_hd.append('gravity_stress3')
        derived_field_list_hd.append('number_density')
        if nscal > 0:
            for n in range(nscal):
                derived_field_list_hd.append('scalar%d' % n)
        self.domain['nscal']=nscal
        self.derived_field_list=derived_field_list_hd+derived_field_list_mhd
        self.derived_field_list_hd=derived_field_list_hd
        self.derived_field_list_mhd=derived_field_list_mhd

    def _set_field_map(self,grid):
        return set_field_map(grid)

    def _read_field(self,file_pointer,field_map):
        return read_field(file_pointer,field_map)

    def _read_grid_data(self,grid,field):
        gd=grid['data']
        if field in gd:
            return

        file=open(grid['filename'],'rb')

        fm=self.domain['field_map']
        nx1=grid['Nx'][0]
        nx2=grid['Nx'][1]
        nx3=grid['Nx'][2]

        if field == 'face_centered_B1': nx1=nx1+1
        if field == 'face_centered_B2': nx2=nx2+1
        if field == 'face_centered_B3': nx3=nx3+1

        nvar=fm[field]['nvar']
        var = self._read_field(file,fm[field])
        if nvar == 1:
            var.shape = (nx3, nx2, nx1)
        else:
            var.shape = (nx3, nx2, nx1, nvar)
        file.close()
        grid['data'][field]=var
        if nvar == 3: self._set_vector_field(grid,field)

    def _get_grid_data(self,grid,field):
        gd=grid['data']
        if field in gd:
            return gd[field]
        elif field in self.field_list:
            self._read_grid_data(grid,field)
            return gd[field]
        elif field in self.derived_field_list:
            data=self._get_derived_field(grid,field)
            return data
        else:
            print('{} is not supported'.format(field))

    def _set_vector_field(self,grid,vfield):
        gd=grid['data']
        gd[vfield+'1'] = gd[vfield][:,:,:,0]
        gd[vfield+'2'] = gd[vfield][:,:,:,1]
        gd[vfield+'3'] = gd[vfield][:,:,:,2]

    def _get_derived_field(self,grid,field):
        gd=grid['data']
        if field in gd:
            return gd[field]
        elif field.startswith('velocity'):
            self._read_grid_data(grid,'velocity')
            if field == 'velocity_magnitude':
                v1=gd['velocity1']
                v2=gd['velocity2']
                v3=gd['velocity3']
                vmag=np.sqrt(v1**2+v2**2+v3**2)
                return vmag
            else: return gd[field]
        elif field.startswith('magnetic_field'):
            self._read_grid_data(grid,'magnetic_field')
            return gd[field]
        elif field.startswith('number_density'):
            self._read_grid_data(grid,'density')
            return gd['density']
        elif field.startswith('kinetic_energy'):
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'velocity')
            den=gd['density']
            v1=gd['velocity1']
            v2=gd['velocity2']
            v3=gd['velocity3']
            if field == 'kinetic_energy1': return 0.5*den*v1**2
            if field == 'kinetic_energy2': return 0.5*den*v2**2
            if field == 'kinetic_energy3': return 0.5*den*v3**2
        elif field.startswith('momentum'):
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'velocity')
            den=gd['density']
            v1=gd['velocity1']
            v2=gd['velocity2']
            v3=gd['velocity3']
            if field == 'momentum1': return den*v1
            if field == 'momentum2': return den*v2
            if field == 'momentum3': return den*v3
        elif field.startswith('magnetic_energy'):
            self._read_grid_data(grid,'magnetic_field')
            B1=gd['magnetic_field1']
            B2=gd['magnetic_field2']
            B3=gd['magnetic_field3']
            if field == 'magnetic_energy1': return B1**2/(8*np.pi)
            if field == 'magnetic_energy2': return B2**2/(8*np.pi)
            if field == 'magnetic_energy3': return B3**2/(8*np.pi)
        elif field.startswith('magnetic_pressure'):
            self._read_grid_data(grid,'magnetic_field')
            B1=gd['magnetic_field1']
            B2=gd['magnetic_field2']
            B3=gd['magnetic_field3']
            if field == 'magnetic_pressure': return (B1**2+B2**2+B3**2)/(8*np.pi)
        elif field.startswith('plasma_beta'):
            vfield='magnetic_field'
            self._read_grid_data(grid,'pressure')
            self._read_grid_data(grid,vfield)
            B1=gd[vfield+'1']
            B2=gd[vfield+'2']
            B3=gd[vfield+'3']
            press=gd['pressure']
            if field == 'plasma_beta': return press*(8.0*np.pi)/(B1**2+B2**2+B3**2)
        elif field.startswith('alfven_velocity'):
            vfield='magnetic_field'
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,vfield)
            den=gd['density']
            B1=gd[vfield+'1']
            B2=gd[vfield+'2']
            B3=gd[vfield+'3']
            if field == 'alfven_velocity1': return B1/np.sqrt(4*np.pi*den)
            if field == 'alfven_velocity2': return B2/np.sqrt(4*np.pi*den)
            if field == 'alfven_velocity3': return B3/np.sqrt(4*np.pi*den)
        elif field.startswith('sound_speed'):
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'pressure')
            den=gd['density']
            press=gd['pressure']
            return np.sqrt(press/den)
        elif field.startswith('temperature'):
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'pressure')
            den=gd['density']
            press=gd['pressure']
            cs2=press/den
            T1=cs2*((au.km/au.s)**2*ac.m_p/ac.k_B).cgs.value
            return coolftn.get_temp(T1)
        elif field.startswith('T1'):
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'pressure')
            den=gd['density']
            press=gd['pressure']
            return press/den*((au.km/au.s)**2*ac.m_p/ac.k_B).cgs.value
        elif field.startswith('potential'):
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'gravitational_potential')
            den=gd['density']
            pot=gd['gravitational_potential']
            return -den*pot
        elif field.startswith('magnetic_stress'):
            vfield='magnetic_field'
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,vfield)
            B1=gd[vfield+'1']
            B2=gd[vfield+'2']
            B3=gd[vfield+'3']
            if field == 'magnetic_stress1': return B2*B3/(4*np.pi)
            if field == 'magnetic_stress2': return B1*B3/(4*np.pi)
            if field == 'magnetic_stress3': return B1*B2/(4*np.pi)
            return B1*B2/(4*np.pi)
        elif field.startswith('reynold_stress'):
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'velocity')
            den=gd['density']
            v1=gd['velocity1']
            v2=gd['velocity2']
            v3=gd['velocity3']
            if field == 'reynold_stress1': return den*v2*v3
            if field == 'reynold_stress2': return den*v1*v3
            if field == 'reynold_stress3': return den*v1*v2
            return den*v1*v2
        elif field.startswith('gravity_stress'):
            self._read_grid_data(grid,'gravitational_potential')
            phi=gd['gravitational_potential']
            dx=grid['dx']
            g1,g2,g3=gradient(phi,dx)
            if field == 'gravity_stress1': return g2*g3/4/np.pi
            if field == 'gravity_stress2': return g1*g3/4/np.pi
            if field == 'gravity_stress3': return g1*g2/4/np.pi
            return  g1*g2/4/np.pi
        elif field.startswith('scalar'):
            scal = field[6:]
            self._read_grid_data(grid,'density')
            self._read_grid_data(grid,'specific_scalar'+scal)
            den=gd['density']
            sscal=gd['specific_scalar'+scal]
            return sscal*den

    def _set_data_array(self,field,dnx):
        fm=self.domain['field_map']

        if field in self.field_list:
            if fm[field]['nvar']==3:
                data=np.empty((dnx[2],dnx[1],dnx[0],3),dtype=fm[field]['dtype'])
            else:
                data=np.empty((dnx[2],dnx[1],dnx[0]),dtype=fm[field]['dtype'])
            if field == 'face_centered_B1':
                data=np.empty((dnx[2],dnx[1],dnx[0]+1),dtype=fm[field]['dtype'])
            if field == 'face_centered_B2':
                data=np.empty((dnx[2],dnx[1]+1,dnx[0]),dtype=fm[field]['dtype'])
            if field == 'face_centered_B3':
                data=np.empty((dnx[2]+1,dnx[1],dnx[0]),dtype=fm[field]['dtype'])
        elif field in self.derived_field_list:
            data=np.empty((dnx[2],dnx[1],dnx[0]),dtype=fm['density']['dtype'])
        else:
            print('{} is not in this file'.format(field))
            print('supported fields are {}'.format(self.field_list))
        return data

    def _get_slab_grid(self,slab=1,verbose=False):
        if slab > self.NGrids[2]:
            print(("%d is lareger than %d" % (slab,self,NGrids[2])))
        NxNy=self.NGrids[0]*self.NGrids[1]
        gidx, = np.where(slab == self.gid/NxNy+1)
        grids = []
        for i in gidx:
            grids.append(self.grids[i])
        if verbose: print(("XY slab from z=%g to z=%g" % (grids[0]['left_edge'][2],grids[0]['right_edge'][2])))
        return grids

    def read_all_data(self,field,slab=False,verbose=False):
        #fm=self.grids[0]['field_map']
        fm=self.domain['field_map']
        dnx=np.copy(self.domain['Nx'])
        if slab:
            dnx[2]=self.grids[0]['Nx'][2]
            grids = self._get_slab_grid(slab=slab,verbose=verbose)
        else:
            grids = self.grids
        data = self._set_data_array(field,dnx)
        for g in grids:
            gis=np.copy(g['is'])
            if slab: gis[2]=0
            gnx=np.copy(g['Nx'])
            gie=gis+gnx
            gd=self._get_grid_data(g,field)
            if field in self.field_list and fm[field]['nvar']==3:
                data[gis[2]:gie[2],gis[1]:gie[1],gis[0]:gie[0],:]=gd
            else:
                if gie[0] == dnx[0] and field == 'face_centered_B1':
                    data[gis[2]:gie[2],gis[1]:gie[1],gis[0]:gie[0]+1]=gd
                elif gie[1] == dnx[1] and field == 'face_centered_B2':
                    data[gis[2]:gie[2],gis[1]:gie[1]+1,gis[0]:gie[0]]=gd
                elif gie[2] == dnx[2] and field == 'face_centered_B3':
                    data[gis[2]:gie[2]+1,gis[1]:gie[1],gis[0]:gie[0]]=gd
                else:
                    gd=gd[0:gnx[2],0:gnx[1],0:gnx[0]]
                    data[gis[2]:gie[2],gis[1]:gie[1],gis[0]:gie[0]]=gd

        return data

def set_field_map(grid):
    file=open(grid['filename'],'rb')
    file.seek(0,2)
    eof = file.tell()
    offset = grid['data_offset']
    file.seek(offset)

    field_map={}

    if 'Nx' in grid: Nx=grid['Nx']

    while offset < eof:

        line=file.readline()
        sp = line.strip().split()
        if len(sp) == 0:
            line=file.readline()
            sp = line.strip().split()
        #print line,sp,len(sp)
        field=sp[1].decode('utf-8')
        field_map[field] = {}
        field_map[field]['read_table']=False

        if b"SCALARS" in line:
            tmp=file.readline()
            field_map[field]['read_table']=True
            field_map[field]['nvar'] = 1
        elif b"VECTORS" in line:
            field_map[field]['nvar'] = 3
        else:
            print(('Error: '+sp[0] + ' is unknown type'))
            raise TypeError

        field_map[field]['offset']=offset
        field_map[field]['ndata']=field_map[field]['nvar']*grid['ncells']
        if field == 'face_centered_B1':
            field_map[field]['ndata']=(Nx[0]+1)*Nx[1]*Nx[2]
        elif field == 'face_centered_B2':
            field_map[field]['ndata']=Nx[0]*(Nx[1]+1)*Nx[2]
        elif field == 'face_centered_B3':
            field_map[field]['ndata']=Nx[0]*Nx[1]*(Nx[2]+1)

        if sp[2]==b'int': dtype='i'
        elif sp[2]==b'float': dtype='f'
        elif sp[2]==b'double': dtype='d'
        field_map[field]['dtype']=dtype
        field_map[field]['dsize']=field_map[field]['ndata']*struct.calcsize(dtype)
        file.seek(field_map[field]['dsize'],1)
        offset = file.tell()
        tmp=file.readline()
        if len(tmp)>1: file.seek(offset)
        else: offset = file.tell()

    #grid['field_map'] = field_map
    #grid['data']={}
    return field_map

def read_field(file_pointer,field_map):
    ndata=field_map['ndata']
    dtype=field_map['dtype']
    file_pointer.seek(field_map['offset'])
    file_pointer.readline() # HEADER
    if field_map['read_table']: file_pointer.readline()
    data = file_pointer.read(field_map['dsize'])
    var = np.asarray(struct.unpack('>'+ndata*dtype,data))

    return var

def read_starvtk(starfile,time_out=False):
    file=open(starfile,'rb')
    star = {}
    star['filename']=starfile
    star['read_field'] = None
    star['read_type'] = None
    while star['read_field'] is None:
        star['data_offset']=file.tell()
        line = file.readline()
        parse_line(line, star)

    time=star['time']
    nstar=star['nstar']
    #print nstar
    fm=set_field_map(star)
    id=read_field(file,fm['star_particle_id'])
    mass=read_field(file,fm['star_particle_mass'])
    age=read_field(file,fm['star_particle_age'])
    pos=read_field(file,fm['star_particle_position']).reshape(nstar,3)
    vel=read_field(file,fm['star_particle_velocity']).reshape(nstar,3)
    misc_keys=fm.keys()
    misc_keys.remove('star_particle_id')
    misc_keys.remove('star_particle_mass')
    misc_keys.remove('star_particle_age')
    misc_keys.remove('star_particle_position')
    misc_keys.remove('star_particle_velocity')
    misc_data={}

    for f in misc_keys:
        misc_data[f]=read_field(file,fm[f])

    file.close()
    star=[]
    for i in range(nstar):
        star.append({})

    for i in range(nstar):
        star_dict = star[i]
        star_dict['id']=id[i]
        star_dict['mass']=mass[i]
        star_dict['age']=age[i]
        star_dict['v1']=vel[i][0]
        star_dict['v2']=vel[i][1]
        star_dict['v3']=vel[i][2]
        star_dict['x1']=pos[i][0]
        star_dict['x2']=pos[i][1]
        star_dict['x3']=pos[i][2]
        star_dict['time']=time
        for f in misc_keys:
            keyname=f.replace('star_particle_','').replace('[','').replace(']','')
            star_dict[keyname]=misc_data[f][i]

    if time_out:
        return time,pd.DataFrame(star)
    else:
        return pd.DataFrame(star)

def gradient(phi,dx):
    Nx=phi.shape

    g1=np.empty(Nx)
    g2=np.empty(Nx)
    g3=np.empty(Nx)

    g1[:,:,1:-1]=(phi[:,:,2:]-phi[:,:,:-2])/dx[0]/2.0
    g1[:,:,0 ]=(phi[:,:,1 ]-phi[:,:,0 ])/dx[0]
    g1[:,:,-1]=(phi[:,:,-1]-phi[:,:,-2])/dx[0]

    g2[:,1:-1,:]=(phi[:,2:,:]-phi[:,:-2,:])/dx[1]/2.0
    g2[:,0 ,:]=(phi[:,1 ,:]-phi[:,0 ,:])/dx[1]
    g2[:,-1,:]=(phi[:,-1,:]-phi[:,-2,:])/dx[1]

    g3[1:-1,:,:]=(phi[2:,:,:]-phi[:-2,:,:])/dx[2]/2.0
    g3[0 ,:,:]=(phi[1 ,:,:]-phi[0 ,:,:])/dx[2]
    g3[-1,:,:]=(phi[-1,:,:]-phi[-2,:,:])/dx[2]

    return g1,g2,g3
