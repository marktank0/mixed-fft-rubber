MixedFFT is a FFT-based solver to calculate mechanical response for RVE. It contains two calculators, one of which is standard strain-based FFT-Galerkin solver, another one of which is mixed one with strain-pressure as unknowns 



# Problem definition

To use the two solvers, it should define the problems at first. Two .txt files are needed. 

The first one is about the geometry of RVE, called "phase.txt". In this file, 0.0 and 1.0 are listed in one column as

```
0.0
1.0
0.0
0.0
1.0
...
```

It is a one column reshaped version of a 3-dimension phase. Note that a 63\* 63\* 63 phase means a 250047 elements for one column. 

The last one is about the material parameters and charge, called "charge.txt" as

```python
#first two lines: model:---0)model num 1) p1.. 2)p2...
#First line; Phase 0(matrix): Model used - E_modulus - Poisson ratio - Rest is padding
1.0	6.0	0.5	0.0	0.0	0.0	0.0	0.0	0.0
#Second line; Phase 1(filler): Model used - E_modulus - Poisson ratio - Rest is padding
1.0	2.6	0.0	0.0	0.0	0.0	0.0	0.0	0.0
#charge dF: so here; strain in F11 for 50% strain
0.5	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
#(Charge type) P-1 or F-0: 0 = control this component by Fij, 1 = control this component by average Pij
0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	1.0
```

The second line corresponds to phase 0.0; The third line is to phase 1.0. For the second line and third line, 1.0 means the constitutive model 1, which is in "fg/constitutive/1.py" or "fg/constitutive_incompressible/1.py". 6.0 and 0.5 are two material parameters. In the case of model 1.0, 6.0 is the Young's module and 0.5 is the Poisson's ratio. Fifth line is the charge, and 7th is the type, see comments after "#"

Material model 1.0 is Neo-Hookean.

After preparing these two files, put them in one folder (we call it "problem path" in this introduction, e.g. "./problem_path/")

# Run calculators

It is better to write a script in the folder containing "fg/". 

1. import fft solver (standard strain-based FFT-Galerkin) or mxfft solver (mixed FFT-Galerkin) as

   ```python
   from fg.fft import *
   ```

   or

   ```python
   from fg.mxfft import *
   ```

   

2. create a solver instance of FFTsolver class, the grid number should be given when initiating. 63 for example

   ```python
   path = "problem_path/"
   prob = FFTsolver(path, N=63)
   ```

3. run the solver. Note that, it should be given a list of ratios of charge. For example, `[1.0]` means only one step, 100% of the charge; `[0.25,0.25,0.25,0.25]` means 4 steps, each step increases 25% of the charge. 

   ```python
   incre_list=[1.0]
   prob.calculate(incre_list=incre_list,savemodel="normal")
   ```

   savemodel is a not yet finished. Just write as it is. After calculation, results of average 1st PK stress and deformation gradient are stored in  the problem path ("./problem_path/") as "P.txt" and "F.txt"

several examples are given as "traction.py", "seres_shear_mix.py", "series_shear_std.py", and "foam2.py"



# On the visualization

The visualization part is not contained in current version, due to a license problem. Users can add it by himself.

For example, if you have downloaded "pyevtk" from https://github.com/pyscience-projects/pyevtk (Note that the original website is https://bitbucket.org/pauloh/pyevtk), you can put the folder in "./fg/" as "./fg/pyevtk" where "evtk.py", "hl.py", "vtk.py" and "xml.py" are in.

add code blocks in "./fg/fft.py"  or "./fg/mxfft.py"

```python

#for forming vtk
from .pyevtk.hl import imageToVTK
#
JJ = np.zeros([N,N,N])
            for x,y,z in itertools.product(range(N),repeat=3):
                ff = F[:,:,x,y,z]
                jj = np.linalg.det(ff)
                JJ[x,y,z] = jj
            #save Fs Ps and all F,P,phase and J
            Allfield = {"P11":P[0,0,:,:,:],"P12":P[0,1,:,:,:],"P13":P[0,2,:,:,:],"P21":P[1,0,:,:,:],\
            "P22":P[1,1,:,:,:],"P23":P[1,2,:,:,:],"P31":P[2,0,:,:,:],"P32":P[2,1,:,:,:],\
            "P33":P[2,2,:,:,:],\
            "F11":F[0,0,:,:,:],"F12":F[0,1,:,:,:],"F13":F[0,2,:,:,:],"F21":F[1,0,:,:,:],\
            "F22":F[1,1,:,:,:],"F23":F[1,2,:,:,:],"F31":F[2,0,:,:,:],"F32":F[2,1,:,:,:],\
            "F33":F[2,2,:,:,:],\
            "phase": phase, "J":JJ}
            imagepath = os.path.join(self.path, "results")
            #imageToVTK(self.path+"results", cellData=Allfield)
            imageToVTK(imagepath, cellData=Allfield)
```

after line 260 in fft.py or line 364 in mxfft.py, after 

```python
        #-------------------------------post 
        print("finish!")
```

Then a .vti file is created after calculation, which can be opened in ParaView.



As pyevtk is GPL licensed, this module is not contained in current MIT licensed FFT solver. Further direct use of vtk tools is to develop.  







