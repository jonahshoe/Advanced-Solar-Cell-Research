import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

sq = 1
multijunction = 1
multiexciton = 1

### Set Joules to 1 to run the script in Joule units, eV to 1 to run in eV units. ###
### If both are set to 1, the script will default to eV units. ###

Joules = 0 #run the script in Joules units
eV = 1 #run the script in eV units
q = 1.602176634e-19 #in C

if Joules == 1:
    h = 6.62607015e-34 #in J*s units
    k = 1.380649e-23 #in J/K

if eV == 1:
    h = 6.62607015e-34/q #in eV*s units
    k = 1.380649e-23/q #in eV/K

c = 299792458 #in m/s
Tsun = 5762 #sun temp in K
Tc = 300 #ambient temp in K
g = 2*np.pi/h**3/c**2
qg = q*g
f = (6.957/1495.98)**2 #(radius of sun / distance to sun)^2
C = 1 #concentration factor

### Import the AM 1.5 Spectrum (from nrel) ###
spectrum = np.asarray(pd.read_csv('am1.5spectrum.txt',skiprows=2,sep='\t'))
wavelength = spectrum[:,0] #in nanometers
radiance = spectrum[:,3] #in W/s/m^2
energies = h*c*(wavelength/1e9)**(-1) #The 1e9 factor is to convert wavelength to meters. energies has same units as h

### Reverse the arrays to put the data values in order of increasing energy / decreasing wavelength ###
wavelength = wavelength[::-1]
radiance = radiance[::-1]
energies = energies[::-1]


def func(x):
    return x**2/(np.exp(x)-1)

def func2(x):
    return x/(np.exp(x)-1)

def func3(x):
    if np.abs(x) > 1e-2:
        return 1/(np.exp(x)-1)
    elif np.abs(x) <= 1e-2:
        return 1/x #taylor series approx for small x


def Q(egLower,egUpper,T):
    """This function calculates the photon flux for energies from egLower to egUpper at temperature T using the quad() integration function from scipy.integrate. The units of egLower and egUpper can be either eV or J. The integral's variables have been changed to make the integral return a unitless value. The units of the returned value will be in whatever energy units k is in."""
    xgLower = egLower/(k*T)
    xgUpper = egUpper/(k*T)
    integral1 = integrate.quad(func,xgLower,xgUpper,limit=10000)
    return (k*T)**3*integral1[0]


def func4(E,T,V):
    return E**2/(np.exp((E-q*V)/(k*T))-1)

def Q_V(Eg,T,V):
    """This function calculates the recombination rate for a solar cell with bandgap energy Eg operating at temperature T and voltage V. The integral will return the cube of whatever units k is in."""
    if Eg > q*V:
        integral1 = integrate.quad(func4,Eg,np.inf,limit=10000,args=(Tc,V))
        return integral1[0]
    else:
        print('Error: Voltage is larger than the bandgap!')
        return 0


def realsolar(wl,P,eg2,eg1):
    """This function calculates the photon flux for the AM 1.5 solar spectrum imported in line 31. The units of the returned value are in photons/m^2/s. wl is the wavelength data in nm, P is the radiance/energy flux data in W/m^2/nm, and Eg is the bandgap energy of the material. The wavelength is converted to energy using E = hc/wavelength (this will be in units of Joules regardless of what option was chosen at the beginning of the script so that it will cancel with the units of P). The energy flux P is then divided by the energy to get the photon flux, ie, W/m^2/nm == J/s/m^2/nm, so (W/m^2/nm)/J == number of photons/s/m^2/nm. Integration across wavelength then produces units of photons/s/m^2. Since it already has s and m^2 in the denominator, it doesn't need to be multiplied by g later."""
    sum = 0
    if Joules == 1:
        energies = h*c*(wavelength/1e9)**(-1)
    if eV == 1:
        energies = h*c*(wavelength/1e9)**(-1)*q
    fluxes = P/energies
    for i in range(len(wl)):
        if i+1 < len(wl):
            wavelengthtemp = 0.5*(wl[i]+wl[i+1])
            if wavelengthtemp <= h*c/eg2*1e9 and wavelengthtemp >= h*c/eg1*1e9: #the 1e9 factor is to convert the RHS to nm
                dwl = np.abs((wl[i+1]-wl[i]))
                sum += 0.5*(fluxes[i]+fluxes[i+1])*dwl
    return sum

def realpower(wl,P):
    """This function calculates the total incident power per m^2 from the AM 1.5 solar spectrum by integrating the radiance P, in units of W/m^2/nm, over all wavelengths. """
    sum = 0
    for i in range(len(wl)):
        if i+1 < len(wl):
            dwl = np.abs((wl[i+1]-wl[i]))
            sum += 0.5*(P[i]+P[i+1])*dwl
    return sum


if Joules == 1:
    eg = np.linspace(0.2,3,200)*q #in Joules
    P_in = 930 #page 8 of Solar Energy document -- for AM 1.5, intensity is approximately 930 W/m^2
elif eV == 1:
    eg = np.linspace(0.2,3,200)
    P_in = 930/q


if sq == 1:
    vcount = 1000 #length of voltage array

    efficiencies = np.zeros([len(eg),vcount])
    efficienciesReal = np.zeros([len(eg),vcount])
    maxEfficiencies = np.zeros_like(eg)
    maxEfficienciesReal = np.zeros_like(eg)
    J = np.zeros([len(eg),vcount])
    JReal = np.zeros([len(eg),vcount])


    for i in range(len(eg)):
        if Joules == 1:
            v = np.linspace(0,eg[i]-0.1,vcount)
        if eV == 1:
            v = np.linspace(0,eg[i]-0.1,vcount)/q #convert from Volts == Joules/Coulomb to eV/Coulomb
        term1 = f*C*Q(eg[i],np.inf,Tsun)
        term1Real = realsolar(wavelength,radiance,eg[i],np.inf)
        term2 = (1-f*C)*Q(eg[i],np.inf,Tc)
        for j in range(vcount):
            ## check to see if the sign of J has changed (but give the first two iterations an exemption) ##
            if j < 2 or J[i,j-2]*J[i,j-1] > 0:
                term3 = Q_V(eg[i],Tc,v[j])
                J[i,j] = qg*(term1+term2-term3)
            else:
                J[i,j] = 0

            if j < 2 or JReal[i,j-2]*JReal[i,j-1] > 0:
                term3 = Q_V(eg[i],Tc,v[j])
                JReal[i,j] = q*term1Real+qg*(term2-term3)
            else:
                JReal[i,j] = 0

            efficiencies[i,j] = J[i,j]*v[j]/P_in
            efficienciesReal[i,j] = JReal[i,j]*v[j]/P_in
        ## efficiencies and efficienciesReal are the efficiency values for all voltage values for just the current bandgap energy. The max needs to be taken to get the maximum possible efficiency at that bandgap energy ##
        maxEfficiencies[i] = np.max(efficiencies[i])
        maxEfficienciesReal[i] = np.max(efficienciesReal[i])
    plt.plot(eg,maxEfficienciesReal)



if multijunction == 1:
    if Joules == 1:
        eg1 = np.linspace(1.4,3,2)*q #in Joules
        eg2 = 1.1*q #bandgap for silicon
        v = np.linspace(0,3,10) #in Volts
        P_in = 930
        P_inReal = 930
    elif eV == 1:
        eg1 = np.linspace(1.5,1.9,50)
        eg2 = 1.1
        v2 = np.linspace(0,1,250)/q
        P_in = 930/q #page 8 of Goodnick's document, for AM1.5, intensity is approximately 930 W/m^2
        P_inReal = 930/q

    v1count = 250

    j1 = np.zeros([len(eg1),v1count,len(v2)])
    j2 = np.zeros([len(eg1),v1count,len(v2)])

    j1Real = np.zeros([len(eg1),v1count,len(v2)])
    j2Real = np.zeros([len(eg1),v1count,len(v2)])

    maxEfficiencyMJ = []
    maxEfficiencyMJReal = []
    finalDiff = []
    finalDiffReal = []

    for i in range(len(eg1)):
        if Joules == 1:
            v1 = np.linspace(0,eg1[i]-0.1,v1count)
        if eV == 1:
            v1 = np.linspace(0,eg1[i]-0.1,v1count)/q
        index1 = []
        index2 = []
        index1Real = []
        index2Real = []
        efficiency = []
        efficiencyReal = []
        diff = np.zeros_like(v2)
        diffReal = np.zeros_like(v2)

        term1_2 = f*C*Q(eg2,eg1[i],Tsun)
        if np.isnan(term1_2) == True:
            term1_2 = 0
        term1_2Real = realsolar(wavelength,radiance,eg2,eg1[i])
        if np.isnan(term1_2Real) == True:
            term1_2real = 0
        term2_2 = (1-f*C)*Q(eg2,eg1[i],Tc)
        if np.isnan(term2_2) == True:
            term2_2 = 0
        term1_1 = f*C*Q(eg1[i],np.inf,Tsun)
        if np.isnan(term1_1) == True:
            term1_1 = 0
        term1_1Real = realsolar(wavelength,radiance,eg1[i],np.inf)
        if np.isnan(term1_1Real) == True:
            term1_1Real = 0
        term2_1 = (1-f*C)*Q(eg1[i],np.inf,Tc)
        if np.isnan(term2_1) == True:
            term2_1 = 0
        storedDiffs = []
        storedDiffsReal = []
        for j in range(v1count):
            term3_1 = Q_V(eg1[i],Tc,v1[j])
            if np.isnan(term3_1) == True:
                term3_1 = 0
            term4_2 = Q_V(eg1[i],Tc,v1[j])
            if np.isnan(term4_2) == True:
                term4_2 = 0

            for n in range(len(v2)):

                term4_1 = Q_V(eg1[i],Tc,v2[n])
                if np.isnan(term4_1) == True:
                    term4_1 = 0

                j1[i,j,n] = qg*(term1_1+term2_1-term3_1+term4_1)
                if j1[i,j,n] < 0:
                    j1[i,j,n] = 0

                j1Real[i,j,n] = q*term1_1Real+qg*(term2_1-term3_1+term4_1)
                if j1Real[i,j,n] < 0:
                    j1Real[i,j,n] = 0

                term3_2 = Q_V(eg2,Tc,v2[n])
                if np.isnan(term3_2) == True:
                    term3_2 = 0

                j2[i,j,n] = qg*(term1_2+term2_2-term3_2+term4_2)
                if j2[i,j,n] < 0:
                    j2[i,j,n] = 0

                j2Real[i,j,n] = q*term1_2Real+qg*(term2_2-term3_2+term4_2)
                if j2Real[i,j,n] < 0:
                    j2Real[i,j,n] = 0

                diff[n] = np.abs(j1[i,j,n]-j2[i,j,n])
                diffReal[n] = np.abs(j1Real[i,j,n] - j2Real[i,j,n])

                javg = (j1[i,j,n]+j2[i,j,n])/2
                if javg != 0 and np.isnan(diff[n]) == False:
                    if diff[n]/javg < 0.1:
                        index1.append(j)
                        index2.append(n)
                        storedDiffs.append(diff[n]/(np.abs(j1[i,j,n]+j2[i,j,n])/2))

                javgReal = (j1Real[i,j,n]+j2Real[i,j,n])/2
                if javgReal != 0 and np.isnan(diffReal[n]) == False:
                    if diffReal[n]/javgReal < 0.1:
                        index1Real.append(j)
                        index2Real.append(n)
                        storedDiffsReal.append(diffReal[n]/(np.abs(j1Real[i,j,n]+j2Real[i,j,n])/2))

        if len(index1) > 0:
            bestDiff = np.min(storedDiffs)
            bestIndex = np.argmin(storedDiffs)
            j1temp = j1[i,index1[bestIndex],index2[bestIndex]]
            j2temp = j2[i,index1[bestIndex],index2[bestIndex]]
            jtemp = 0.5*(np.abs(j1temp)+np.abs(j2temp))
            v1temp = v1[index1[bestIndex]]
            v2temp = v2[index2[bestIndex]]
            vtemp = v1temp + v2temp
            efficiency.append(jtemp*vtemp/P_in)
            maxEfficiencyMJ.append(np.max(np.asarray(efficiency)))
            finalDiff.append(storedDiffs[bestIndex])
        else:
            maxEfficiencyMJ.append(np.nan)

        if len(index1Real) > 0:
            for b in range(len(index1Real)):
                j1tempReal = j1Real[i,index1Real[b],index2Real[b]]
                j2tempReal = j2[i,index1Real[b],index2Real[b]]
                jtempReal = 0.5*(np.abs(j1tempReal)+np.abs(j2tempReal))
                v1tempReal = v1[index1Real[b]]
                v2tempReal = v2[index2Real[b]]
                vtempReal = v1tempReal + v2tempReal
                efficiencyReal.append(jtempReal*vtempReal/P_in)
            maxEfficiencyMJReal.append(np.max(np.asarray(efficiencyReal)))
            finalDiffReal.append(storedDiffsReal[b])
        else:
            maxEfficiencyMJReal.append(np.nan)

    plt.plot(eg1,maxEfficiencyMJReal)





if multiexciton == 1:
    if Joules == 1:
        eg = np.linspace(0.1,3,100)*q #in Joules
        v = np.linspace(0,3,2500) #in Volts
    elif eV == 1:
        eg = np.linspace(0.1,3,100)
        v = np.linspace(0,3,200)/q #in eV/Coulomb instead of Joules/Coulomb

    if Joules == 1:
        P_in = 930
    if eV == 1:
        P_in = 930/q #page 8 of Goodnick's document, for AM1.5, intensity is approximately 930 W/m^2



    meg_cap = 6 #maximum number of electrons that can be generated by a high-energy photon

    def efficiency(meg):
        n = np.zeros([len(eg),len(v)])
        n_real = np.zeros([len(eg),len(v)])
        nmax = np.zeros_like(eg)
        nmax_real = np.zeros_like(eg)
        j1 = np.zeros([len(eg),len(v)])
        j1real = np.zeros([len(eg),len(v)])


        for i in range(len(eg)):
            term1 = np.zeros([meg])
            term1_real = np.zeros([meg])
            term2 = np.zeros([meg])
            for u in range(meg):
                if u+1 < meg:
                    term1[u] = (u+1)*f*C*Q((u+1)*eg[i],(u+2)*eg[i],Tsun)
                    term1_real[u] = (u+1)*realsolar(wavelength,radiance,(u+1)*eg[i],(u+2)*eg[i])
                    term2[u] = (u+1)*(1-f*C)*Q((u+1)*eg[i],(u+2)*eg[i],Tc)
                elif u+1 == meg:
                    term1[u] = (u+1)*f*C*Q((u+1)*eg[i],np.inf,Tsun)
                    term1_real[u] = (u+1)*realsolar(wavelength,radiance,(u+1)*eg[i],np.inf)
                    term2[u] = (u+1)*(1-f*C)*Q((u+1)*eg[i],np.inf,Tc)

            for j in range(len(v)):
                term3 = Q_V2(eg[i],np.inf,Tc,v[j])
                # for u in range(meg):
                #     if u+1 < meg:
                #         term3[u] = (u+1)*Q_V2((u+1)*eg[i],(u+2)*eg[i],Tc,v[j])
                #     elif u+1 == meg:
                #         term3[u] = (u+1)*Q_V2((u+1)*eg[i],np.inf,Tc,v[j])

                if j < 2 or j1[i,j-2]*j1[i,j-1] > 0:
                    for u in range(meg):
                        j1[i,j] += qg*(term1[u]+term2[u])
                    j1[i,j] -= qg*term3
                else:
                    j1[i,j] = 0

                if j < 2 or j1real[i,j-2]*j1real[i,j-1] > 0:
                    for u in range(meg):
                        j1real[i,j] += q*term1_real[u]+qg*term2[u]
                    j1real[i,j] -= qg*term3
                else:
                    j1real[i,j] = 0

                n[i,j] = j1[i,j]*v[j]/P_in
                n_real[i,j] = j1real[i,j]*v[j]/P_in
            nmax[i] = np.max(n[i])
            nmax_real[i] = np.max(n_real[i])
        return nmax,nmax_real


    nmax_array = np.zeros([meg_cap,len(eg)])
    nmax_real_array = np.zeros([meg_cap,len(eg)])
    for jk in range(meg_cap):
        nmax_array[jk],nmax_real_array[jk] = efficiency(jk+1)

plt.show()


















e
