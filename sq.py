import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

########## Set sq to 1 to calculate standard Shockley-Queisser limit
########## Set multijunction to 1 to calculate efficiency for two-layer tandem with Si base
########## Set multiexciton to 1 to calculate efficiency for multi-exciton generation
########## Set multijunctionMEG to 1 to calculate efficiency for two-layer tandem with MEG
sq = 1
multijunction = 1
multiexciton = 1 #set meg_cap to desired number
meg_cap = 6 #maximum number of electrons that can be generated by a high-energy photon for multiexciton
multijunctionMEG = 0 #multijunctionMEG is still a work in progress


blackbody = 0 #set to 1 to calculate efficiencies for a blackbody spectrum source
am1pt5 = 1 #set to 1 to calculate efficiencies using AM 1.5 spectrum from nrel


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


if am1pt5 == 1:
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
    """This function calculates the photon flux for energies from egLower
    to egUpper at temperature T using the quad() integration function from scipy.integrate.
    The units of egLower and egUpper can be either eV or J. The integral's variables have been
    changed to make the integral return a unitless value. The units of the returned value will
    be in whatever energy units k is in."""
    xgLower = egLower/(k*T)
    xgUpper = egUpper/(k*T)
    integral1 = integrate.quad(func,xgLower,xgUpper,limit=10000)
    return (k*T)**3*integral1[0]


def func4(E,T,V):
    return E**2/(np.exp((E-q*V)/(k*T))-1)

def Q_V(Eg,T,V):
    """This function calculates the recombination rate for a solar cell with bandgap energy Eg
    operating at temperature T and voltage V. The integral will return the cube of whatever units k is in."""
    if Eg > q*V:
        integral1 = integrate.quad(func4,Eg,np.inf,limit=10000,args=(Tc,V))
        return integral1[0]
    else:
        print('Error: Voltage is larger than the bandgap!')
        print('Eg = ', Eg)
        print('q*V = ', q*V)
        return 0


def realsolar(wl,P,eg2,eg1):
    """This function calculates the photon flux for the AM 1.5 solar spectrum imported in line 31.
    he units of the returned value are in photons/m^2/s. wl is the wavelength data in nm, P is the
    radiance/energy flux data in W/m^2/nm, and Eg is the bandgap energy of the material. The wavelength
    is converted to energy using E = hc/wavelength (this will be in units of Joules regardless of what
    option was chosen at the beginning of the script so that it will cancel with the units of P).
    The energy flux P is then divided by the energy to get the photon flux, ie, W/m^2/nm == J/s/m^2/nm,
    so (W/m^2/nm)/J == number of photons/s/m^2/nm. Integration across wavelength then produces units of
    photons/s/m^2. Since it already has s and m^2 in the denominator, it doesn't need to be multiplied by g later."""
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
    """This function calculates the total incident power per m^2 from the AM 1.5 solar spectrum by
    integrating the radiance P, in units of W/m^2/nm, over all wavelengths. """
    sum = 0
    for i in range(len(wl)):
        if i+1 < len(wl):
            dwl = np.abs((wl[i+1]-wl[i]))
            sum += 0.5*(P[i]+P[i+1])*dwl
    return sum



fignum = 1 #for plotting


if sq == 1:
    if Joules == 1:
        eg = np.linspace(0.2,3,200)*q #in Joules
        P_in = 930 #page 8 of Solar Energy document -- for AM 1.5, intensity is approximately 930 W/m^2
    elif eV == 1:
        eg = np.linspace(0.2,3,200)
        P_in = 930/q
    vcount = 1000 #length of voltage array

    efficiencies = np.zeros([len(eg),vcount])
    efficienciesReal = np.zeros([len(eg),vcount])
    maxEfficiencies = np.zeros_like(eg)
    maxEfficienciesReal = np.zeros_like(eg)
    J = np.zeros([len(eg),vcount])
    JReal = np.zeros([len(eg),vcount])

    ##### Decoder: term1 = absorbed solar flux for blackbody spectrum
    ##### term1Real = absorbed solar flux for AM 1.5 spectrum
    ##### term2 = absorbed ambient flux
    ##### term3 = lost recombined radiation
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
        ## efficiencies and efficienciesReal are the efficiency values for all
        ## voltage values for just the current bandgap energy. The max needs to
        ## be taken to get the maximum possible efficiency at that bandgap energy
        maxEfficiencies[i] = np.max(efficiencies[i])
        maxEfficienciesReal[i] = np.max(efficienciesReal[i])
    plt.figure(fignum)
    fignum += 1
    plt.plot(eg,maxEfficienciesReal)
    plt.title('Single Junction Shockley-Queisser Efficiency Limit')
    plt.xlabel('Bandgap Energy (eV)')
    plt.ylabel('Efficiency')
    print('Finished with Single Junction!')






if multijunction == 1:
    ### Take a smaller step size in regions further from the predicted optimal bandgap energy
    ### by concatenating three arrays with different step sizes for different energy ranges
    egcount1 = 20
    egcount2 = 50
    egcount3 = 30
    egcount = egcount1 + egcount2 + egcount3
    eglimit1 = 1.5
    eglimit2 = 1.6
    eglimit3 = 1.8
    eglimit4 = 2.0
    if Joules == 1:
        eg1a = np.linspace(eglimit1,eglimit2,egcount1)
        eg1b = np.linspace(eglimit2,eglimit3,egcount2)
        eg1c = np.linspace(eglimit3,eglimit4,egcount3)
        eg1 = np.concatenate((eg1a,eg1b,eg1c),axis=0)*q
        eg2 = 1.1*q #bandgap for silicon
        P_in = 930
    elif eV == 1:
        eg1a = np.linspace(eglimit1,eglimit2,egcount1)
        eg1b = np.linspace(eglimit2,eglimit3,egcount2)
        eg1c = np.linspace(eglimit3,eglimit4,egcount3)
        eg1 = np.concatenate((eg1a,eg1b,eg1c),axis=0)
        eg2 = 1.1
        P_in = 930/q #page 8 of Goodnick's document, for AM1.5, intensity is approximately 930 W/m^2

    vcount = 100 #resolution of voltage arrays

    if blackbody == 1:
        j1 = np.zeros([egcount,vcount,vcount])
        j2 = np.zeros([egcount,vcount,vcount])
        maxEfficiencyMJ = []

    if am1pt5 == 1:
        j1Real = np.zeros([egcount,vcount,vcount])
        j2Real = np.zeros([egcount,vcount,vcount])
        maxEfficiencyMJReal = []


    for i in range(egcount):
        if Joules == 1:
            v = np.linspace(0,eg1[i]-0.1,vcount)
            v2 = np.linspace(0,eg2-0.1,vcount)
        if eV == 1:
            v = np.linspace(0,eg1[i]-0.1,vcount)/q
            v2 = np.linspace(0,eg2-0.1,vcount)/q

        ##### This next part calculates all the terms that don't depend on the
        ##### voltages. For each, there is a check to make sure that a Not-a-number
        ##### wasn't returned by any of the integrals. This should have been mostly
        ##### fixed by taking the voltage arrays to the bandgap energies - 0.1 eV,
        ##### but the NaN checks only slow down the computation a little, and save
        ##### a lot of potential headaches, so I left them in.

        ##### Decoder: term1_1 = absorbed solar flux for top layer (first term, first layer).
        ##### term1_2 = absorbed solar flux for silicon base layer (first term, second layer).
        ##### term2_1 = absorbed ambient flux for top layer (second term, first layer).
        ##### term2_2 = absorbed ambient flux for silicon layer (second term, second layer).
        ##### term3_1 = recombined radiation given off by top layer (third term, first layer).
        ##### term3_2 = recomined radiation given off by silicon layer (third term, second layer).
        ##### term4_1 = absorbed flux given off by the silicon layer, absorbed by the top layer (fourth term, first layer).
        ##### term4_2 = absorbed flux given off by the top layer, absorbed by the bottom layer (fourth term, second layer).

        if blackbody == 1:
            index1 = []
            index2 = []
            efficiency = []
            diff = np.zeros_like(v)
            term1_2 = f*C*Q(eg2,eg1[i],Tsun)
            if np.isnan(term1_2) == True:
                term1_2 = 0
            term1_1 = f*C*Q(eg1[i],np.inf,Tsun)
            if np.isnan(term1_1) == True:
                term1_1 = 0
            storedDiffs = []

        if am1pt5 == 1:
            index1Real = []
            index2Real = []
            efficiencyReal = []
            diffReal = np.zeros_like(v)
            term1_2Real = realsolar(wavelength,radiance,eg2,eg1[i])
            if np.isnan(term1_2Real) == True:
                term1_2real = 0
            term1_1Real = realsolar(wavelength,radiance,eg1[i],np.inf)
            if np.isnan(term1_1Real) == True:
                term1_1Real = 0
            storedDiffsReal = []

        term2_2 = (1-f*C)*Q(eg2,eg1[i],Tc)
        if np.isnan(term2_2) == True:
            term2_2 = 0

        term2_1 = (1-f*C)*Q(eg1[i],np.inf,Tc)
        if np.isnan(term2_1) == True:
            term2_1 = 0

        for j in range(vcount):
            term3_1 = Q_V(eg1[i],Tc,v[j])
            if np.isnan(term3_1) == True:
                term3_1 = 0
            term4_2 = Q_V(eg1[i],Tc,v[j])
            if np.isnan(term4_2) == True:
                term4_2 = 0

            for n in range(vcount):

                term4_1 = Q_V(eg1[i],Tc,v2[n])
                if np.isnan(term4_1) == True:
                    term4_1 = 0

                term3_2 = Q_V(eg2,Tc,v2[n])
                if np.isnan(term3_2) == True:
                    term3_2 = 0

                ##### This calculates both layers' current densities, then takes their
                ##### difference and their average. Any pair of current densities
                ##### that satisfies the requirement that their difference divided
                ##### by their average is less than the parameter "threshold" is
                ##### accepted, and their indices are stored in index1 and index2.

                threshold = 0.1

                if blackbody == 1:
                    j1[i,j,n] = qg*(term1_1+term2_1-term3_1+term4_1)
                    if j1[i,j,n] < 0:
                        j1[i,j,n] = 0
                    j2[i,j,n] = qg*(term1_2+term2_2-term3_2+term4_2)
                    if j2[i,j,n] < 0:
                        j2[i,j,n] = 0
                    diff[n] = np.abs(j1[i,j,n]-j2[i,j,n])
                    javg = (j1[i,j,n]+j2[i,j,n])/2
                    if javg != 0 and np.isnan(diff[n]) == False:
                        if diff[n]/javg < threshold:
                            index1.append(j)
                            index2.append(n)
                            storedDiffs.append(diff[n]/(np.abs(j1[i,j,n]+j2[i,j,n])/2))

                if am1pt5 == 1:
                    j1Real[i,j,n] = q*term1_1Real+qg*(term2_1-term3_1+term4_1)
                    if j1Real[i,j,n] < 0:
                        j1Real[i,j,n] = 0
                    j2Real[i,j,n] = q*term1_2Real+qg*(term2_2-term3_2+term4_2)
                    if j2Real[i,j,n] < 0:
                        j2Real[i,j,n] = 0
                    diffReal[n] = np.abs(j1Real[i,j,n] - j2Real[i,j,n])
                    javgReal = (j1Real[i,j,n]+j2Real[i,j,n])/2
                    if javgReal != 0 and np.isnan(diffReal[n]) == False:
                        if diffReal[n]/javgReal < threshold:
                            index1Real.append(j)
                            index2Real.append(n)
                            storedDiffsReal.append(diffReal[n]/(np.abs(j1Real[i,j,n]+j2Real[i,j,n])/2))

        ##### This next part calculates the efficiency for each of the pairs of
        ##### current densities that passes the prior threshold criteria. The
        ##### average of the current densities and sum of the voltages are used
        ##### to calculate the produced power. The efficiency and efficiencyReal
        ##### lists are appended with the efficiencies for all of the pairs of
        ##### current densities. The best of these is stored in the maxEfficiencyMJ
        ##### and maxEfficiencyMJReal arrays and the efficiency and efficiencyReal
        ##### arrays will be cleared for the next bandgap energy value.

        if blackbody == 1:
            if len(index1) > 0:
                for b in range(len(index1)):
                    j1temp = j1[i,index1[bestIndex],index2[bestIndex]]
                    j2temp = j2[i,index1[bestIndex],index2[bestIndex]]
                    jtemp = 0.5*(np.abs(j1temp)+np.abs(j2temp))
                    v1temp = v[index1[bestIndex]]
                    v2temp = v2[index2[bestIndex]]
                    vtemp = v1temp + v2temp
                    efficiency.append(jtemp*vtemp/P_in)
                maxEfficiencyMJ.append(np.max(np.asarray(efficiency)))
            else:
                maxEfficiencyMJ.append(np.nan)

        if am1pt5 == 1:
            if len(index1Real) > 0:
                for b in range(len(index1Real)):
                    j1tempReal = j1Real[i,index1Real[b],index2Real[b]]
                    j2tempReal = j2Real[i,index1Real[b],index2Real[b]]
                    jtempReal = 0.5*(np.abs(j1tempReal)+np.abs(j2tempReal))
                    v1tempReal = v[index1Real[b]]
                    v2tempReal = v2[index2Real[b]]
                    vtempReal = v1tempReal + v2tempReal
                    efficiencyReal.append(jtempReal*vtempReal/P_in)
                maxEfficiencyMJReal.append(np.max(np.asarray(efficiencyReal)))
            else:
                maxEfficiencyMJReal.append(np.nan)
    plt.figure(fignum)
    fignum += 1
    if blackbody == 1:
        plt.plot(eg1,maxEfficiencyMJ)
    if am1pt5 == 1:
        plt.plot(eg1,maxEfficiencyMJReal)
    plt.title('Two-Layer Tandem Solar Cell with Silicon Base Layer')
    plt.xlabel('Bandgap Energy of Top Layer (eV)')
    plt.ylabel('Efficiency')
    print('Finished with the Multi-Junction!')




if multiexciton == 1:
    vcount = 200
    egcount = 100
    eglower = 0.5
    egupper = 2

    if Joules == 1:
        eg = np.linspace(eglower,egupper,egcount)*q #in Joules
        P_in = 930
    elif eV == 1:
        eg = np.linspace(eglower,egupper,egcount)
        P_in = 930/q #page 8 of Goodnick's document, for AM1.5, intensity is approximately 930 W/m^2


    def efficiency(meg):
        """This function takes a maximum value of excitons generated per photon
        and calculates the efficiency of a cell in which every photon
        generates the max number of excitons possible for its energy, up to the
        value of meg. It returns an array of maximum efficiencies as a function
        of bandgap energy."""
        if blackbody == 1:
            n = np.zeros([egcount,vcount])
            nmax = np.zeros_like(eg)
            j1 = np.zeros([egcount,vcount])

        if am1pt5 == 1:
            n_real = np.zeros([egcount,vcount])
            nmax_real = np.zeros_like(eg)
            j1real = np.zeros([egcount,vcount])

        for i in range(egcount):
            if Joules == 1:
                v = np.linspace(0,eg[i]-0.1,vcount)
            if eV == 1:
                v = np.linspace(0,eg[i]-0.1,vcount)/q

            if blackbody == 1:
                term1 = np.zeros([meg])
            if am1pt5 == 1:
                term1_real = np.zeros([meg])
            term2 = np.zeros([meg])

            ##### This for loop will repeat for every possible number of excitons
            ##### generated. Term 1, the incident solar photon flux, and term 2,
            ##### the ambient incident photon flux, have been split into a series of
            ##### integrals, ie, integral(Eg to inf) is replaced with integral(Eg to 2*Eg)
            ##### + 2*integral(2*Eg to 3*Eg) + 3*integral(3*Eg to 4*Eg) + ... up to
            ##### the maximum provided by meg. This loop cycles through every interval,
            ##### from Eg to 2Eg, 2Eg to 3Eg, etc, and stores the result of each integral
            ##### inside of term1[u], term1_real, term2[u].
            for u in range(meg):
                if u+1 < meg:
                    if blackbody == 1:
                        term1[u] = (u+1)*f*C*Q((u+1)*eg[i],(u+2)*eg[i],Tsun)
                    if am1pt5 == 1:
                        term1_real[u] = (u+1)*realsolar(wavelength,radiance,(u+1)*eg[i],(u+2)*eg[i])
                    term2[u] = (u+1)*(1-f*C)*Q((u+1)*eg[i],(u+2)*eg[i],Tc)
                elif u+1 == meg:
                    if blackbody == 1:
                        term1[u] = (u+1)*f*C*Q((u+1)*eg[i],np.inf,Tsun)
                    if am1pt5 == 1:
                        term1_real[u] = (u+1)*realsolar(wavelength,radiance,(u+1)*eg[i],np.inf)
                    term2[u] = (u+1)*(1-f*C)*Q((u+1)*eg[i],np.inf,Tc)

            ##### This for loop cycles through all of the voltage values, calculates
            ##### term3 for each, then loops through meg again to add all of the
            ##### term1[u], term2[u], etc integrals found in the previous loop.
            ##### The efficiency is calculated for each current density found.
            for j in range(vcount):
                term3 = Q_V(eg[i],Tc,v[j])

                if blackbody == 1:
                    if j < 2 or j1[i,j-2]*j1[i,j-1] > 0:
                        for u in range(meg):
                            j1[i,j] += qg*(term1[u]+term2[u])
                        j1[i,j] -= qg*term3
                    else:
                        j1[i,j] = 0
                    n[i,j] = j1[i,j]*v[j]/P_in

                if am1pt5 == 1:
                    if j < 2 or j1real[i,j-2]*j1real[i,j-1] > 0:
                        for u in range(meg):
                            j1real[i,j] += q*term1_real[u]+qg*term2[u]
                        j1real[i,j] -= qg*term3
                    else:
                        j1real[i,j] = 0
                    n_real[i,j] = j1real[i,j]*v[j]/P_in

            if blackbody == 1:
                nmax[i] = np.max(n[i])
            if am1pt5 == 1:
                nmax_real[i] = np.max(n_real[i])

        if blackbody == 1:
            if am1pt5 == 1:
                return nmax, nmax_real
            if am1pt5 == 0:
                return nmax
        else:
            return nmax_real

    ##### This next loop will cycle through meg_cap and will return the max
    ##### efficiencies for each possible max number of excitons less than or equal
    ##### to meg_cap. Ie, for meg_cap = 2, the max efficiencies for single-exciton
    ##### generation *and* double-exciton generation will be returned and graphed.
    if blackbody == 1:
        nmax_array = np.zeros([meg_cap,egcount])
    if am1pt5 == 1:
        nmax_real_array = np.zeros([meg_cap,egcount])
    plt.figure(fignum)
    fignum += 1
    for jk in range(meg_cap):
        if blackbody == 1:
            if am1pt5 == 1:
                nmax_array[jk],nmax_real_array[jk] = efficiency(jk+1)
                plt.plot(eg,nmax_real_array[jk])
                plt.plot(eg,nmax_real[jk])
            if am1pt5 == 0:
                nmax_array[jk] = efficiency(jk+1)
                plt.plot(eg,nmax_real[jk])
        else:
            nmax_real_array[jk] = efficiency(jk+1)
            plt.plot(eg,nmax_real_array[jk])
    plt.title('Shockley-Queisser Limit with Multi-Exciton Generation for ' + str(meg_cap) + ' Max Exciton(s) per Photon')
    plt.xlabel('Bandgap Energy (eV)')
    plt.ylabel('Efficiency')
    print('Finished with the multi-exciton generation!')
plt.show()


if multijunctionMEG == 1:
    egcount1 = 30
    egcount2 = 120
    egcount3 = 50
    egcount = egcount1 + egcount2 + egcount3
    eglimit1 = 1.2
    eglimit2 = 1.55
    eglimit3 = 1.85
    eglimit4 = 2.5
    if Joules == 1:
        eg1a = np.linspace(eglimit1,eglimit2,egcount1)
        eg1b = np.linspace(eglimit2,eglimit3,egcount2)
        eg1c = np.linspace(eglimit3,eglimit4,egcount3)
        eg1 = np.concatenate((eg1a,eg1b,eg1c),axis=0)*q
        eg2 = 1.1*q #bandgap for silicon
        P_in = 930
        P_inReal = 930
    elif eV == 1:
        eg1a = np.linspace(eglimit1,eglimit2,egcount1)
        eg1b = np.linspace(eglimit2,eglimit3,egcount2)
        eg1c = np.linspace(eglimit3,eglimit4,egcount3)
        eg1 = np.concatenate((eg1a,eg1b,eg1c),axis=0)
        eg2 = 1.1
        P_in = 930/q #page 8 of Goodnick's document, for AM1.5, intensity is approximately 930 W/m^2
        P_inReal = 930/q

    vcount = 100

    if blackbody == 1:
        j1 = np.zeros([egcount,vcount,vcount])
        j2 = np.zeros([egcount,vcount,vcount])
        maxEfficiencyMJ = []
        finalDiff = []

    if am1pt5 == 1:
        j1Real = np.zeros([egcount,vcount,vcount])
        j2Real = np.zeros([egcount,vcount,vcount])
        maxEfficiencyMJReal = []
        finalDiffReal = []


    for i in range(egcount):
        if Joules == 1:
            v = np.linspace(0,eg1[i]-0.1,vcount)
            v2 = np.linspace(0,eg2-0.1,vcount)
        if eV == 1:
            v = np.linspace(0,eg1[i]-0.1,vcount)/q
            v2 = np.linspace(0,eg2-0.1,vcount)/q

        if blackbody == 1:
            index1 = []
            index2 = []
            efficiency = []
            diff = np.zeros_like(v)
            for u in range(meg):
                if u+1 < meg:
                    if blackbody == 1:
                        term1_1[u] = (u+1)*f*C*Q((u+1)*eg[i],(u+2)*eg[i],Tsun)
                    if am1pt5 == 1:
                        term1_real[u] = (u+1)*realsolar(wavelength,radiance,(u+1)*eg[i],(u+2)*eg[i])
                    term2[u] = (u+1)*(1-f*C)*Q((u+1)*eg[i],(u+2)*eg[i],Tc)
                elif u+1 == meg:
                    if blackbody == 1:
                        term1[u] = (u+1)*f*C*Q((u+1)*eg[i],np.inf,Tsun)
                    if am1pt5 == 1:
                        term1_real[u] = (u+1)*realsolar(wavelength,radiance,(u+1)*eg[i],np.inf)
                    term2[u] = (u+1)*(1-f*C)*Q((u+1)*eg[i],np.inf,Tc)
            term1_2 = f*C*Q(eg2,eg1[i],Tsun)
            if np.isnan(term1_2) == True:
                term1_2 = 0
            term1_1 = f*C*Q(eg1[i],np.inf,Tsun)
            if np.isnan(term1_1) == True:
                term1_1 = 0
            storedDiffs = []

        if am1pt5 == 1:
            index1Real = []
            index2Real = []
            efficiencyReal = []
            diffReal = np.zeros_like(v)
            term1_2Real = realsolar(wavelength,radiance,eg2,eg1[i])
            if np.isnan(term1_2Real) == True:
                term1_2real = 0
            term1_1Real = realsolar(wavelength,radiance,eg1[i],np.inf)
            if np.isnan(term1_1Real) == True:
                term1_1Real = 0
            storedDiffsReal = []

        term2_2 = (1-f*C)*Q(eg2,eg1[i],Tc)
        if np.isnan(term2_2) == True:
            term2_2 = 0

        term2_1 = (1-f*C)*Q(eg1[i],np.inf,Tc)
        if np.isnan(term2_1) == True:
            term2_1 = 0

        for j in range(vcount):
            term3_1 = Q_V(eg1[i],Tc,v[j])
            if np.isnan(term3_1) == True:
                term3_1 = 0
            term4_2 = Q_V(eg1[i],Tc,v[j])
            if np.isnan(term4_2) == True:
                term4_2 = 0

            for n in range(vcount):

                term4_1 = Q_V(eg1[i],Tc,v2[n])
                if np.isnan(term4_1) == True:
                    term4_1 = 0

                term3_2 = Q_V(eg2,Tc,v2[n])
                if np.isnan(term3_2) == True:
                    term3_2 = 0

                if blackbody == 1:
                    j1[i,j,n] = qg*(term1_1+term2_1-term3_1+term4_1)
                    if j1[i,j,n] < 0:
                        j1[i,j,n] = 0
                    j2[i,j,n] = qg*(term1_2+term2_2-term3_2+term4_2)
                    if j2[i,j,n] < 0:
                        j2[i,j,n] = 0
                    diff[n] = np.abs(j1[i,j,n]-j2[i,j,n])
                    javg = (j1[i,j,n]+j2[i,j,n])/2
                    if javg != 0 and np.isnan(diff[n]) == False:
                        if diff[n]/javg < 0.1:
                            index1.append(j)
                            index2.append(n)
                            storedDiffs.append(diff[n]/(np.abs(j1[i,j,n]+j2[i,j,n])/2))

                if am1pt5 == 1:
                    j1Real[i,j,n] = q*term1_1Real+qg*(term2_1-term3_1+term4_1)
                    if j1Real[i,j,n] < 0:
                        j1Real[i,j,n] = 0
                    j2Real[i,j,n] = q*term1_2Real+qg*(term2_2-term3_2+term4_2)
                    if j2Real[i,j,n] < 0:
                        j2Real[i,j,n] = 0
                    diffReal[n] = np.abs(j1Real[i,j,n] - j2Real[i,j,n])
                    javgReal = (j1Real[i,j,n]+j2Real[i,j,n])/2
                    if javgReal != 0 and np.isnan(diffReal[n]) == False:
                        if diffReal[n]/javgReal < 0.1:
                            index1Real.append(j)
                            index2Real.append(n)
                            storedDiffsReal.append(diffReal[n]/(np.abs(j1Real[i,j,n]+j2Real[i,j,n])/2))






        if blackbody == 1:
            if len(index1) > 0:
                bestDiff = np.min(storedDiffs)
                bestIndex = np.argmin(storedDiffs)
                j1temp = j1[i,index1[bestIndex],index2[bestIndex]]
                j2temp = j2[i,index1[bestIndex],index2[bestIndex]]
                jtemp = 0.5*(np.abs(j1temp)+np.abs(j2temp))
                v1temp = v[index1[bestIndex]]
                v2temp = v2[index2[bestIndex]]
                vtemp = v1temp + v2temp
                efficiency.append(jtemp*vtemp/P_in)
                maxEfficiencyMJ.append(np.max(np.asarray(efficiency)))
                finalDiff.append(storedDiffs[bestIndex])
            else:
                maxEfficiencyMJ.append(np.nan)

        if am1pt5 == 1:
            if len(index1Real) > 0:
                for b in range(len(index1Real)):
                    j1tempReal = j1Real[i,index1Real[b],index2Real[b]]
                    j2tempReal = j2Real[i,index1Real[b],index2Real[b]]
                    jtempReal = 0.5*(np.abs(j1tempReal)+np.abs(j2tempReal))
                    v1tempReal = v[index1Real[b]]
                    v2tempReal = v2[index2Real[b]]
                    vtempReal = v1tempReal + v2tempReal
                    efficiencyReal.append(jtempReal*vtempReal/P_in)
                maxEfficiencyMJReal.append(np.max(np.asarray(efficiencyReal)))
                finalDiffReal.append(storedDiffsReal[b])
            else:
                maxEfficiencyMJReal.append(np.nan)
    plt.figure(fignum)
    fignum += 1
    if blackbody == 1:
        plt.plot(eg1,maxEfficiencyMJ)
    if am1pt5 == 1:
        plt.plot(eg1,maxEfficiencyMJReal)
    print('Finished with the Multi-Junction!')
















e
