# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table, Column
from astropy.table import join as joinTables
from fitmag import *
import os
import emcee
from datetime import datetime
from os.path import join, abspath, dirname, exists
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import argparse
from time import time as time_
import sys
from .version import __version__

__all__ = ['BV_to_Teff', 'Teff_to_BV', 
           'gr_to_Teff', 'Teff_to_gr', 
           'gi_to_Teff', 'Teff_to_gi', 
           'gJ_to_Teff', 'Teff_to_gJ', 
           'gH_to_Teff', 'Teff_to_gH', 
           'gK_to_Teff', 'Teff_to_gK', 
           'VK_to_Teff', 'Teff_to_VK', 
           'VW3_to_Teff', 'Teff_to_VW3', 
           'VIc_to_Teff', 'Teff_to_VIc', 
           'gr_to_BV', 'gr_to_Kp', 'BK_to_S_V', 'BK_to_S_B',
           'BV_to_BTVT', 'gri_to_Kp',
           'gi_to_T',
           'read_data', 'add_mags', 'gTeff_to_magTable',
           'synth_binary', 'synth_triple', 'ms_interpolator']

#-----------------------------------------------------------------------------

def BV_to_Teff(BV):
    if (BV >= -0.02) & (BV <= 1.73):
        return np.polyval([4009, -24072, 57338, -68940, 44350, -17443, 9552],BV)
    else:
        return np.nan

def Teff_to_BV(Teff):
    for r in np.roots([4009, -24072, 57338, -68940, 44350, -17443, 9552-Teff]):
        if (r.imag == 0) & (r.real >= -0.02) & (r.real <= 1.73):
            return r.real
    return np.nan

def Teff_to_VRj(Teff):
    for r in np.roots([-1302.5, 5579, -9272, 9335-Teff]):
        if (r.imag == 0) & (r.real >=  0.00) & (r.real <= 1.69):
            return r.real
    return np.nan

def Teff_to_VIj(Teff):
    for r in np.roots([-245.1, 1884, -5372,  9189-Teff]):
        if (r.imag == 0) & (r.real >= -0.03) & (r.real <= 3.12):
            return r.real
    return np.nan

def gr_to_Teff(gr):
    if (gr >= -0.23) & (gr <= 1.40):
        return np.polyval([-1332.9, 3750, -5570, 7526],gr)
    else:
        return np.nan

def Teff_to_gr(Teff):
    for r in np.roots([-1332.9, 3750, -5570, 7526-Teff]):
        if (r.imag == 0) & (r.real >= -0.23) & (r.real <= 1.40):
            return r.real
    return np.nan

def gi_to_Teff(gi):
    if (gi >= -0.43) & (gi <= 2.78):
        return np.polyval([-153.9, 1112, -3356, 7279],gi)
    else:
        return np.nan

def Teff_to_gi(Teff):
    for r in np.roots([-153.9, 1112, -3356, 7279-Teff]):
        if (r.imag == 0) & (r.real >= -0.43) & (r.real <= 2.78):
            return r.real
    return np.nan

def gJ_to_Teff(gJ):
    if (gJ >= -0.02) & (gJ <= 5.06):
        return np.polyval([-51.5, 623, -2933, 8759],gJ)
    else:
        return np.nan

def Teff_to_gJ(Teff):
    for r in np.roots([-51.5, 623, -2933, 8759-Teff]):
        if (r.imag == 0) & (r.real >= -0.02) & (r.real <= 5.06):
            return r.real
    return np.nan

def gH_to_Teff(gH):
    if (gH >= -0.12) & (gH <= 5.59):
        return np.polyval([-32.3, 432, -2396, 8744],gH)
    else:
        return np.nan

def Teff_to_gH(Teff):
    for r in np.roots([-32.3, 432, -2396, 8744-Teff]):
        if (r.imag == 0) & (r.real >= -0.12) & (r.real <= 5.59):
            return r.real
    return np.nan

def gK_to_Teff(gK):
    if (gK >= -0.01) & (gK <= 5.86):
        return np.polyval([-25.8, 365, -2178, 8618],gK)
    else:
        return np.nan

def Teff_to_gK(Teff):
    for r in np.roots([-25.8, 365, -2178, 8618-Teff]):
        if (r.imag == 0) & (r.real >= -0.01) & (r.real <= 5.86):
            return r.real
    return np.nan

def VK_to_Teff(VK):
    if (VK >= -0.15) & (VK <= 5.04):
        return np.polyval([-49.2, 606, -2968, 9030],VK)
    else:
        return np.nan

def Teff_to_VK(Teff):
    for r in np.roots([-49.2, 606, -2968, 9030-Teff]):
        if (r.imag == 0) & (r.real >= -0.15) & (r.real <= 5.04):
            return r.real
    return np.nan

def VW3_to_Teff(VW3):
    if (VW3 >=  0.76) & (VW3 <= 5.50):
        return np.polyval([-45.3, 602, -3005, 9046],VW3)
    else:
        return np.nan

def Teff_to_VW3(Teff):
    for r in np.roots([-45.3, 602, -3005, 9046-Teff]):
        if (r.imag == 0) & (r.real >=  0.76) & (r.real <= 5.50):
            return r.real
    return np.nan

def VIc_to_Teff(VIc):
    if (VIc >= -0.02) & (VIc <= 2.77):
        return np.polyval([-536.9, 3333, -7360, 9440],VIc)
    else:
        return np.nan

def Teff_to_VIc(Teff):
    for r in np.roots([-536.9, 3333, -7360, 9440-Teff]):
        if (r.imag == 0) & (r.real >= -0.02) & (r.real <= 2.77):
            return r.real
    return np.nan

#-----------------------------------------------------------------------------

def gr_to_BV(g, r, return_sigma=False):
  # Lupton (2005) transformation 
  # See https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
  # sigma_B = 0.0107
  # sigma_V = 0.0054
  if (return_sigma) :
      return 0.0107, 0.0054
  else :
      return g + 0.3130*(g - r) + 0.2271, g - 0.5784*(g - r) - 0.0038

#-----------------------------------------------------------------------------

def gr_to_Kp(g, r):
  # See https://keplergo.arc.nasa.gov/CalibrationZeropoint.shtml
  if ( g-r <= 0.8):
    Kp = 0.2*g + 0.8*r
  else:
    Kp = 0.1*g + 0.9*r
  return Kp

#-----------------------------------------------------------------------------

def gri_to_Kp(g, r, i):
  # From Brown 2011AJ....142..112B
  if ( g-r <= 0.3):
    Kp = 0.25*g + 0.75*r
  else:
    Kp = 0.3*g + 0.7*i
  return Kp

#-----------------------------------------------------------------------------

def gi_to_T(g, i):
  # From Stassun et al., 2018AJ....156..102S
  T = i -  0.00206*(g-i)**3 -  0.02370*(g-i)**2 + 0.00573*(g-i) - 0.3078
  return T

#-----------------------------------------------------------------------------

def BK_to_S_V(BK):
  # Surface brightness relation from Table 5 of Graczyk, 2017
    return 2.625 + 0.959*BK

#-----------------------------------------------------------------------------

def BK_to_S_B(BK):
  # Surface brightness relation from Table 5 of Graczyk, 2017
    return 2.640 + 1.252*BK

#-----------------------------------------------------------------------------

def add_mags(m1, m2, e1=None, e2=None, m3=None, e3=None):
    if m3 is None :
        m =  -2.5*np.log10(10**(-0.4*m1)+10**(-0.4*m2))
        if (e1 is None) and (e2 is None) :
            return m
        e = np.hypot(
            0.5*(add_mags(m1-e1,m2) - add_mags(m1+e1,m2)) ,
            0.5*(add_mags(m1,m2-e2) - add_mags(m1,m2+e2)) )
        return m, e
    else :
        m =  -2.5*np.log10(10**(-0.4*m1)+10**(-0.4*m2) + 10**(-0.4*m3))
        if (e1 is None) and (e2 is None) and (e3 is None) :
            return m
        v = ((0.5*(add_mags(m1-e1,m2,m3=m3)-add_mags(m1+e1,m2,m3=m3)))**2 + 
             (0.5*(add_mags(m1,m2-e2,m3=m3)-add_mags(m1,m2+e2,m3=m3)))**2 +
             (0.5*(add_mags(m1,m2,m3=m3-e3)-add_mags(m1,m2,m3=m3+e3)))**2 )
        e = np.sqrt(v)
        return m, e

#-----------------------------------------------------------------------------

def Ar_to_EBV(A_r):
  """
  Using A(r)/E(B-V) from Fig. 3 of Fiorucci & Munari, 2003A&A...401..781F 
  Returns value =  A_r/2.770 
  """
  return A_r/2.770


def EBV_to_Ar(EBV):
  """
  Using A(r)/E(B-V) from Fig. 3 of Fiorucci & Munari, 2003A&A...401..781F 
  Returns value =  A_r/2.770 
  """
  return EBV*2.770

#-----------------------------------------------------------------------------

def gTeff_to_magTable(g_0, Teff, EBV=0.0):

  A_r = EBV_to_Ar(EBV)

  r_0 = g_0 - Teff_to_gr(Teff)
  B_0, V_0 = gr_to_BV(g_0, r_0)
  V = V_0 + 3.1*EBV
  B = V + (B_0-V_0) + EBV
  B_T, V_T = BV_to_BTVT(B, V)
  g = g_0 + 1.39*A_r
  r = r_0 + A_r
  i_0 = g_0 - Teff_to_gi(Teff) 
  i = i_0 + 0.76*A_r
  Kp = gri_to_Kp(g, r, i)
  T  = gi_to_T(g, i)
  J_0 = g_0 - Teff_to_gJ(Teff) 
  J_J = J_0 + 0.30*A_r # Johnson J
  H_J = g_0 - Teff_to_gH(Teff) + 0.21*A_r  # Johnson H
  K_0 = g_0 - Teff_to_gK(Teff)  # Unreddened Johnson K
  K_J = K_0 + 0.15*A_r 
  # Transformation from Johnson to Bessel & Brett system from
  # 1988PASP..100.1134B
  VK_BB = 0.01 + 0.993 * (V - K_J)
  JH_BB = -0.004 + 1.01*(J_J - H_J)
  JK_BB = 0.01 + 0.99*(J_J - K_J)
  # HK_BB = 0.01 + 0.91*(H_J - K_J)  # Uncertain 
  # Transformation from  Bessel & Brett K to 2MASS Ks from
  # http://www.astro.caltech.edu/%7Ejmc/2mass/v3/transformations/
  K_BB = V - VK_BB 
  Ks = K_BB -0.039 + 0.001*JK_BB 
  JKs = 0.983*JK_BB - 0.018
  JH  = 0.990*JH_BB - 0.049
  # HKs = 0.971*HK_BB + 0.034
  J = JKs + Ks
  H = J - JH 
  # WISE W3 
  W3 = V_0 - Teff_to_VW3(Teff) + 0.13*A_r

  # Extinction from ADPS for Sun-like stars 
  # http://ulisse.pd.astro.it/Astro/ADPS/ADPS2/FileHtml/index_f093.html
  Ic = V_0 - Teff_to_VIc(Teff) + 1.927*EBV
  # Using Bessell 1990 instead of Johnson since most likely dealing with 
  # CCD photometry
  # http://ulisse.pd.astro.it/Astro/ADPS/ADPS2/FileHtml/index_f136.html
  Rj = V_0 - Teff_to_VRj(Teff) + 2.616*EBV
  Ij = V_0 - Teff_to_VIj(Teff) + 1.866*EBV


  bands  = Column(['B', 'V', 'B_T', 'V_T', 'g', 'r', 'i', 'Ic', 
                   'R_J', 'I_J', 'J', 'H', 'Ks', 'W3', 'K_p', 'T'])
  values = Column([B, V, B_T, V_T, g, r, i, Ic, Rj, Ij, J, H, Ks, W3, Kp, T],
          format='%8.4f')
  types  = Column(list('mag' for j in range(len(bands))))
  t =  Table([bands,values,types],names=["band","value","type"])

  S_V = BK_to_S_V(B_0-K_0)
  t.add_row(['V', S_V,'sb2'])
  S_B = BK_to_S_B(B_0-K_0)
  t.add_row(['B', S_B,'sb2'])
  S_Rj = S_V - Teff_to_VRj(Teff)
  t.add_row(['R_J', S_Rj,'sb2'])
  S_Ij = S_V - Teff_to_VIj(Teff)
  t.add_row(['I_J', S_Ij,'sb2'])
  Kp_0 = gri_to_Kp(g_0, r_0, i_0)
  T_0 = gi_to_T(g_0, i_0)
  S_Kp = S_V - V_0 + Kp_0
  S_T  = S_V - V_0 + T_0
  t.add_row(['K_p', S_Kp,'sb2'])
  t.add_row(['T', S_T,'sb2'])

  return t


#-----------------------------------------------------------------------------

def read_data(data_file):
  """
   Read list of apparent magnitudes and other observations from user's data
   file 
  """
  data_table = Table.read(data_file,format='ascii', 
      names=['type','band','value','error','source'])
  assert min(data_table['error']) > 0, (
          "Invalid zero/negative error value in input data table")

  return data_table

#-----------------------------------------------------------------------------

def BV_to_BTVT(B, V):
    # Polynomial fits to Table 2 of Bessell PASP 112,961, 2000.
    pV = np.array([-0.0102072,0.04157612,-0.08908576,0.09318613, 
                   -0.14652784,0.00018468])
    pB1 = np.array([-0.49887625, 0.37480425,-0.2475667 ,-0.0077756 ])
    pB2 = np.array([ 0.04488478,-0.22274173,-0.07598775,-0.01324362])

    BV = B - V
    if np.isscalar(BV):
        if (BV >= -0.219) & (BV <= 1.658) :
            if (BV < 0.5):
                BT = B - np.polyval(pB1,BV)
            else:
                BT = B - np.polyval(pB2,BV)
            VT =   V - np.polyval(pV,BV)
            return BT, VT
        else:
            return np.nan, np.nan
    else:

        BT = np.empty_like(BV)
        BT[:] = np.nan
        i1 = np.where((BV >= -0.219) & (BV < 0.5))
        BT[i1] = B[i1] - np.polyval(pB1,BV[i1])
        i2 = np.where((BV >= 0.5) & (BV <= 1.658))
        BT[i2] = B[i2] - np.polyval(pB2,BV[i2])

        VT = np.empty_like(BV)
        VT[:] = np.nan
        i =  np.where((BV >= -0.219) & (BV <= 1.658))
        VT[i] = V[i] - np.polyval(pV, BV[i])
        return BT, VT



#-----------------------------------------------------------------------------

class ms_interpolator:
  def __init__(self):
  
    # Load intrinsic color sequence data 
    dir_path = dirname(abspath(__file__))
    ms_data_file = join(dir_path,'data','g_Teff_zams_tams.dat')
    ms_data_table = Table.read(ms_data_file,format='ascii')

    # Set up interpolating functions
    self._zams = interp1d(ms_data_table["Teff_ZAMS"],ms_data_table["M_g_ZAMS"],
                          kind='linear')
    self._tams = interp1d(ms_data_table["Teff_TAMS"],ms_data_table["M_g_TAMS"],
                          kind='linear')
  def __call__(self,Teff):
    return self._zams(Teff), self._tams(Teff)

#-----------------------------------------------------------------------------

def synth_binary(g1, Teff1, g2, Teff2, EBV):

  t1 = gTeff_to_magTable(g1, Teff1, EBV)
  t2 = gTeff_to_magTable(g2, Teff2, EBV)
  m = Column(add_mags(t1['value'], t2['value']))
  t = Table([t1['band'],m,t1["value"], t2["value"],t1['type']],
      names=["band","value","value_A","value_B","type"])

  r1 = g1 - Teff_to_gr(Teff1)
  r2 = g2 - Teff_to_gr(Teff2)
  Kp1 = gr_to_Kp(g1, r1)
  Kp2 = gr_to_Kp(g2, r2)
  rat_Kp = 10**(0.4*(Kp1-Kp2)) 
  t.add_row(['K_p', rat_Kp,Kp1,Kp2,'rat'])
  
  i1 = g1 - Teff_to_gi(Teff1)
  i2 = g2 - Teff_to_gi(Teff2)
  T1 = gi_to_T(g1, i1)
  T2 = gi_to_T(g2, i2)
  rat_T = 10**(0.4*(T1-T2)) 
  t.add_row(['T', rat_T,T1,T2,'rat'])
  
  B1,V1 = gr_to_BV(g1, r1)
  B2,V2 = gr_to_BV(g2, r2)
  rat_V = 10**(0.4*(V1-V2)) 
  t.add_row(['V', rat_V,V1,V2,'rat'])

  rat_B = 10**(0.4*(B1-B2)) 
  t.add_row(['B', rat_B,B1,B2,'rat'])

  Rj1 = V1 - Teff_to_VRj(Teff1)
  Rj2 = V2 - Teff_to_VRj(Teff2)
  rat_Rj = 10**(0.4*(Rj1-Rj2)) 
  t.add_row(['R_J', rat_Rj,Rj1,Rj2,'rat'])

  Ij1 = V1 - Teff_to_VIj(Teff1)
  Ij2 = V2 - Teff_to_VIj(Teff2)
  rat_Ij = 10**(0.4*(Ij1-Ij2)) 
  t.add_row(['I_J', rat_Ij,Ij1,Ij2,'rat'])

  for i in range(len(t)):
      if t['type'][i] == 'sb2':
          t['value'][i] = 10**(0.4*(t["value_A"][i]-t["value_B"][i]))

  return  t

#-----------------------------------------------------------------------------

def synth_triple(g1, Teff1, g2, Teff2, g3, Teff3, EBV):

  t1 = gTeff_to_magTable(g1, Teff1, EBV)
  t2 = gTeff_to_magTable(g2, Teff2, EBV)
  t3 = gTeff_to_magTable(g3, Teff3, EBV)
  m = Column(add_mags(t1['value'], t2['value'],m3=t3['value']))
  t = Table([t1['band'],m,t1["value"], t2["value"],t3["value"],t1['type']],
      names=["band","value","value_A","value_B","value_C","type"])

  for i in range(len(t)):
      if t['type'][i] == 'sb2':
          t['value'][i] = 10**(0.4*(t["value_A"][i]-t["value_B"][i]))
      if t['type'][i] == 'mag':
          m_A = t['value_A'][i]
          m_B = t['value_B'][i]
          m_C = t['value_C'][i]
          l_3 = 10**(-0.4*m_C)/(10**(-0.4*m_A) + 10**(-0.4*m_B))
          t.add_row([t['band'][i],l_3,m_A,m_B,m_C,'l_3'])
          rat = 10**(0.4*(m_A-m_B))
          t.add_row([t['band'][i],rat,m_A,m_B,0.0,'rat'])

  return  t

#-----------------------------------------------------------------------------

def weights(t, sig_ext):
    v = t['error']**2 
    v[t['type'] == 'mag'] += sig_ext**2 
    return 1/v

#-----------------------------------------------------------------------------

def chisq1(param, data_table, sig_ext):
    g1, Teff1, EBV = param
    t = gTeff_to_magTable(g1, Teff1, EBV)
    j = joinTables(data_table,t,keys=["band","type"])  
    c = np.sum( (j["value_1"]-j["value_2"])**2 * weights(j, sig_ext))
    return c

#-----------------------------------------------------------------------------

def chisq2(param, data_table, sig_ext):
    g1, Teff1, g2, Teff2, EBV = param
    t= synth_binary(g1, Teff1, g2, Teff2, EBV)
    j = joinTables(data_table,t,keys=["band","type"])  
    c = np.sum( (j["value_1"]-j["value_2"])**2 * weights(j, sig_ext))
    return c

#-----------------------------------------------------------------------------
def chisqd(param, data_table, sig_ext):
    g1, Teff1, dg, Teff2, EBV = param
    g2 = g1 + dg
    t= synth_binary(g1, Teff1, g2, Teff2, EBV)
    j = joinTables(data_table,t,keys=["band","type"])  
    c = np.sum( (j["value_1"]-j["value_2"])**2 * weights(j, sig_ext))
    return c

#-----------------------------------------------------------------------------

def chisq3(param, data_table, sig_ext):
    g1, Teff1, g2, Teff2, g3, Teff3, EBV = param
    t= synth_triple(g1, Teff1, g2, Teff2, g3, Teff3, EBV)
    j = joinTables(data_table,t,keys=["band","type"])  
    c = np.sum((j["value_1"]-j["value_2"])**2 * weights(j, sig_ext))
    return c

#-----------------------------------------------------------------------------

def lnlike1(par, data_table, ebv_map=0, gaussian_ebv=False,
            return_fit=False):
    g1, Teff1, EBV, sig_ext = par 

    if (g1 <= 5) or (g1 > 25) : return -np.inf
    if (Teff1 <= 3450) or (Teff1 > 8600) : return -np.inf
    if (sig_ext < 0)  : return -np.inf
    if (EBV < 0) : return -np.inf
    if gaussian_ebv:
        j = np.where(data_table['type'] == 'ebv')
        v = data_table['value'][j]
        s = data_table['error'][j]
        lnprior = -0.5*np.sum(((v-EBV)/s)**2)
    elif (EBV > ebv_map) : 
        lnprior = -0.5*((EBV-ebv_map)/0.034)**2
    else:
        lnprior = 0.0
        
    t = gTeff_to_magTable(g1, Teff1, EBV)
    j = joinTables(data_table,t,keys=("band","type"))
    wt = weights(j, sig_ext)
    z2 = (j["value_1"]-j["value_2"])**2 * wt
    j.add_column(Column(np.sqrt(z2), name="z",format="%.2f"))
    if return_fit:
        return j
    else:
      lnlike = -0.5*(np.sum(z2 - np.log(wt))) 
      if np.isfinite(lnlike):
          return  lnlike + lnprior
      else:
          return -np.inf

#-----------------------------------------------------------------------------

def lnlike2(par, data_table, ebv_map=0, gaussian_ebv=False,
            return_fit=False):
    g1, Teff1, g2, Teff2, EBV, sig_ext = par 

    if (g1 <= 5) or (g1 > 25) : return -np.inf
    if (g2 <= 5) or (g2 > 25) : return -np.inf
    if (Teff1 <= 3450) or (Teff1 > 8600) : return -np.inf
    if (Teff2 <= 3450) or (Teff2 > 8600) : return -np.inf
    if (sig_ext < 0)  : return -np.inf
    if (EBV < 0) : return -np.inf
    if gaussian_ebv:
        j = np.where(data_table['type'] == 'ebv')
        v = data_table['value'][j]
        s = data_table['error'][j]
        lnprior = -0.5*np.sum(((v-EBV)/s)**2)
    elif (EBV > ebv_map) : 
        lnprior = -0.5*((EBV-ebv_map)/0.034)**2
    else:
        lnprior = 0.0
        
    t= synth_binary(g1, Teff1, g2, Teff2, EBV)
    j = joinTables(data_table,t,keys=("band","type"))
    wt = weights(j, sig_ext)
    z2 = (j["value_1"]-j["value_2"])**2 * wt
    j.add_column(Column(np.sqrt(z2), name="z",format="%.2f"))
    if return_fit:
        return j
    else:
      lnlike = -0.5*(np.sum(z2 - np.log(wt))) 
      if np.isfinite(lnlike):
          return  lnlike + lnprior
      else:
          return -np.inf

#-----------------------------------------------------------------------------

def lnliked(par, data_table, ebv_map=0, ms_interp=None, gaussian_ebv=False,
            return_fit=False):

    g1, Teff1, dg, Teff2, EBV, sig_ext = par 

    g2 = g1 + dg
    if (g1 > g2) : return -np.inf
    if (g1 <= 5) or (g1 > 25) : return -np.inf
    if (g2 <= 5) or (g2 > 25) : return -np.inf
    if (Teff1 <= 3450) or (Teff1 > 8600) : return -np.inf
    if (Teff2 <= 3450) or (Teff2 > 8600) : return -np.inf
    if (sig_ext < 0)  : return -np.inf
    if (EBV < 0) : return -np.inf
    if gaussian_ebv:
        j = np.where(data_table['type'] == 'ebv')
        v = data_table['value'][j]
        s = data_table['error'][j]
        lnprior = -0.5*np.sum(((v-EBV)/s)**2)
    elif (EBV > ebv_map) : 
        lnprior = -0.5*((EBV-ebv_map)/0.034)**2
    else:
        lnprior = 0.0

    if ms_interp is not None:
        g_ZAMS_1, g_TAMS_1 = ms_interp(Teff1)
        g_ZAMS_2, g_TAMS_2 = ms_interp(Teff2)
        if ((g2-g1) < (g_ZAMS_2 - g_ZAMS_1)):
            return -np.inf
        if ((g2-g1) > (g_TAMS_2 - g_TAMS_1)):
            return -np.inf
        
    t= synth_binary(g1, Teff1, g2, Teff2, EBV)
    j = joinTables(data_table,t,keys=("band","type"))
    wt = weights(j, sig_ext)
    z2 = (j["value_1"]-j["value_2"])**2 * wt
    j.add_column(Column(np.sqrt(z2), name="z",format="%.2f"))
    if return_fit:
        return j
    else:
      lnlike = -0.5*(np.sum(z2 - np.log(wt))) 
      if np.isfinite(lnlike):
          return  lnlike + lnprior
      else:
          return -np.inf


#-----------------------------------------------------------------------------

def lnlike3(par, data_table, ebv_map=0, ms_interp=None, gaussian_ebv=False,
            return_fit=False):
    g1, Teff1, g2, Teff2, g3, Teff3, EBV, sig_ext = par

    if (g1 <= 5) or (g1 > 25) : return -np.inf
    if (g2 <= 5) or (g2 > 25) : return -np.inf
    if (g3 <= 5) or (g3 > 25) : return -np.inf
    if (Teff1 <= 3450) or (Teff1 > 8600) : return -np.inf
    if (Teff2 <= 3450) or (Teff2 > 8600) : return -np.inf
    if (Teff3 <= 3450) or (Teff3 > 8600) : return -np.inf
    if (sig_ext < 0)  : return -np.inf
    if (EBV < 0) : return -np.inf
    if gaussian_ebv:
        j = np.where(data_table['type'] == 'ebv')
        v = data_table['value'][j]
        s = data_table['error'][j]
        lnprior = -0.5*np.sum(((v-EBV)/s)**2)
    elif (EBV > ebv_map) :
        lnprior = -0.5*((EBV-ebv_map)/0.033)**2
    else:
        lnprior = 0.0
    if ms_interp is not None:
        g_ZAMS_3, g_TAMS_3 = ms_interp(Teff3)
        if g2 > g1 :
            g_ZAMS_B, g_TAMS_B = ms_interp(Teff2)
            if ((g3-g2) < (g_ZAMS_3 - g_ZAMS_B)):
                return -np.inf
            if ((g3-g2) > (g_TAMS_3 - g_TAMS_B)):
                return -np.inf
        else:
            g_ZAMS_B, g_TAMS_B = ms_interp(Teff1)
            if ((g3-g1) < (g_ZAMS_3 - g_ZAMS_B)):
                return -np.inf
            if ((g3-g1) > (g_TAMS_3 - g_TAMS_B)):
                return -np.inf
            
    t= synth_triple(g1, Teff1, g2, Teff2, g3, Teff3, EBV)
    j = joinTables(data_table,t,keys=("band","type"))
    wt = weights(j, sig_ext)
    z2 = (j["value_1"]-j["value_2"])**2 * wt
    j.add_column(Column(np.sqrt(z2), name="Z",format="%.3f"))
    if return_fit:
        return j
    else:
      lnlike = -0.5*(np.sum(z2 - np.log(wt))) 
      if np.isfinite(lnlike):
          return lnlike + lnprior
      else:
          return -np.inf



#------------------------------------------------------------------------------

def main():

  # Set up command line switched
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Binary/triple star photometry analysis'
  )

  parser.add_argument("phot_file", 
    help='Photometry file'
  )

  parser.add_argument("-b", "--burn-in", 
    default=128, type=int,
    help='Number of burn-in steps'
  )

  parser.add_argument("-w", "--walkers", 
    default=64, type=int,
    help='Number of walkers'
  )

  parser.add_argument("-t", "--threads", 
    default=4, type=int,
    help='Number of threads for emcee'
  )

  parser.add_argument("-m", "--ms-prior", 
    action="store_const",
    const=True,
    default=False,
    help='Impose main-sequence prior'
  )

  parser.add_argument("-n", "--no-progress-meter", 
    action="store_const",
    const=True,
    default=False,
    help='Do not display the progress meter'
  )

  parser.add_argument("-d", "--double", 
    action="store_const",
    const=True,
    default=False,
    help='Fit data for non-eclipsing binary as a double star'
  )

  parser.add_argument("-g", "--gaussian-reddening-prior", 
    action="store_const",
    const=True,
    default=False,
    help='Use Gaussian prior for specified E(B-V) value and error'
  )

  parser.add_argument("-s", "--steps", 
    default=256, type=int,
    help='Number of emcee chain steps for output'
  )

  parser.add_argument("-c", "--chain-file", 
    default='chain.fits',
    help='Output file for chain data'
  )

  parser.add_argument("-f", "--overwrite", 
    action="store_const",
    dest='overwrite',
    const=True,
    default=False,
    help='Force overwrite of existing output files.'
  )

  parser.add_argument("-x", "--initial_sig_ext", 
    default=0.05, type=float,
    help='Initial estimate for sig_ext'
  )

  parser.add_argument("-1", "--initial_teff_1", 
    default=6000, type=int,
    help='Initial estimate for T_eff,1'
  )

  parser.add_argument("-2", "--initial_teff_2", 
    default=5000, type=int,
    help='Initial estimate for T_eff,2'
  )

  parser.add_argument("-3", "--initial_teff_3", 
    default=4000, type=int,
    help='Initial estimate for T_eff,3'
  )

  # Get command line options
  args = parser.parse_args()

  datetime_start = datetime.today()
  print("\nStart fitmag version {} at {:%c}\n".
          format(__version__,datetime_start))

  # Check if output chain file exists 
  if exists(args.chain_file) and not args.overwrite:
      raise IOError("Output chain file exists, use -f option to overwrite")

  data_table = read_data(args.phot_file)
  vals = data_table['value']
  types = data_table['type']
  bands = data_table['band']
  if ('g' in bands):
      g = np.median(vals[np.where((bands == 'g') & (types == 'mag'))])
  else:
      g = 10.0

  if ('rat' in types):
      if args.double:
          raise Exception('Option --double incompatible with input file.')
      lrat = np.median(vals[np.where(
          ((bands == 'K_p') | (bands == 'T') | (bands == 'V')) & 
           (types == 'rat'))])
  else:
      lrat = 0.0

  if ('sb2' in types):
      if args.double:
          raise Exception('Option --double incompatible with input file.')
      sb2 = np.median(vals[np.where(
          ((bands == 'K_p') | (bands == 'V') | (bands == 'T')) & 
           (types == 'sb2'))])
  else:
      sb2 = 0.0

  print("\nRead {} lines from {}\n".format(len(data_table),args.phot_file))

  try:
      ebv_map=np.median(vals[np.where(types == 'ebv')])
  except:
      raise Exception('No E(B-V) in input file')

  sig_ext = args.initial_sig_ext
  assert sig_ext >= 0, "Invalid negative initial sig_ext value"


  print(" Calculating least-squares solution...")
  if ('l_3' in data_table['type']):
      if args.double:
          raise Exception('Option --double incompatible with input file.')

      l_3 = np.median(vals[np.where((
           (bands == 'K_p') | (bands == 'T') | (bands == 'V')) &
           (types == 'l_3'))])
      g_0 = g - 1.39*ebv_map*2.770
      f_g = 10**(-0.4*g_0)
      f_3 = f_g/(1+1/l_3)
      g_3 = -2.5*np.log10(f_3)
      f_2 = (f_g-f_3)/(1+1/lrat)
      g_2 = -2.5*np.log10(f_2)
      f_1 = f_g - f_2 - f_3
      g_1 = -2.5*np.log10(f_1)
      if (sb2 < 1):
          param_0 = (g_1, args.initial_teff_1,
                     g_2, args.initial_teff_2,  
                     g_3, args.initial_teff_3, ebv_map )
      else:
          param_0 = (g_1, args.initial_teff_2,
                     g_2, args.initial_teff_1, 
                     g_3, args.initial_teff_3, ebv_map )
      bounds = ((g_1-5, g_1+5), (3450, 8600), (g_2-5, g_2+5), (3450, 8600),
                (g_3-5, g_3+5), (3450, 8600), (0.0, 2*ebv_map) )
      soln=minimize(chisq3, param_0, method="L-BFGS-B",bounds=bounds, 
                        args=(data_table, sig_ext))
  elif args.double:
      g_0 = g - 1.39*ebv_map*2.770
      f_g = 10**(-0.4*g_0)
      f_2 = f_g/10
      g_2 = -2.5*np.log10(f_2)
      f_1 = f_g - f_2
      g_1 = -2.5*np.log10(f_1)
      dg = g_2 - g_1
      param_0 = (g_1, args.initial_teff_1, dg, args.initial_teff_2, 
          ebv_map )
      bounds = ((g_1-5, g_1+5), (3450, 8600), (0, 10), (3450, 8600),
                (0.0, 2*ebv_map) )
      soln=minimize(chisqd, param_0, method="L-BFGS-B",bounds=bounds, 
                    args=(data_table, sig_ext))

  elif (lrat > 0):
      g_0 = g - 1.39*ebv_map*2.770
      f_g = 10**(-0.4*g_0)
      f_2 = f_g/(1+1/lrat)
      g_2 = -2.5*np.log10(f_2)
      f_1 = f_g - f_2 
      g_1 = -2.5*np.log10(f_1)
      if (sb2 < 1):
          param_0 = (g_1, args.initial_teff_1, g_2, args.initial_teff_2,
              ebv_map )
      else:
          if (args.initial_teff_1 < args.initial_teff_2):
              param_0 = (g_1, args.initial_teff_1, g_2,
                  args.initial_teff_2, ebv_map )
          else:
              param_0 = (g_1, args.initial_teff_2, g_2, args.initial_teff_1,
                  ebv_map )
      bounds = ((g_1-5, g_1+5), (3450, 8600), (g_2-5, g_2+5), (3450, 8600),
                (0.0, 2*ebv_map) )
      soln=minimize(chisq2, param_0, method="L-BFGS-B",bounds=bounds, 
                    args=(data_table, sig_ext))
  else:
      g_1 = g
      param_0 = (g_1, 5500, ebv_map )
      bounds = ((g_1-5, g_1+5), (3450, 8600), (0.0, 2*ebv_map) )
      soln=minimize(chisq1, param_0, method="L-BFGS-B",bounds=bounds, 
                    args=(data_table, sig_ext))

  if (lrat > 0) :
      print("  g_1     = {:5.2f}".format(soln.x[0]))
      print("  T_eff,1 = {:4.0f} K".format(soln.x[1]))
      print("  g_2     = {:5.2f}".format(soln.x[2]))
      print("  T_eff,2 = {:4.0f} K".format(soln.x[3]))
      if ('l_3' in data_table['type']):
          print("  g_3     = {:5.2f}".format(soln.x[4]))
          print("  T_eff,3 = {:4.0f} K".format(soln.x[5]))
          print("  E(B-V)  = {:6.2f}".format(soln.x[6]))
          e_p = [0.01, 100, 0.01, 100, 0.01, 100, 0.01, 0.001]
      else:
          print("  E(B-V)  = {:6.2f}".format(soln.x[4]))
          e_p = [0.01, 100, 0.01, 100, 0.01, 0.001]
  elif  args.double :
      print("  g_1     = {:5.2f}".format(soln.x[0]))
      print("  T_eff,1 = {:4.0f} K".format(soln.x[1]))
      print("  g_2     = {:5.2f}".format(soln.x[2]+soln.x[0]))
      print("  T_eff,2 = {:4.0f} K".format(soln.x[3]))
      print("  E(B-V)  = {:6.2f}".format(soln.x[4]))
      e_p = [0.01, 100, 0.01, 100, 0.01, 0.001]
  else:
      print("  g_0     = {:5.2f}".format(soln.x[0]))
      print("  T_eff   = {:4.0f} K".format(soln.x[1]))
      print("  E(B-V)  = {:6.2f}".format(soln.x[2]))
      e_p = [0.01, 100, 0.01, 0.001]
  print("  chi-squared = {:.2f}".format(soln.fun))
  print("  Ndf = {}".format(len(data_table)-len(param_0)))
  print("  sigma_ext = {}".format(sig_ext))

  n_steps = args.burn_in + args.steps
  n_threads = args.threads
  p_0 = np.append(soln.x, sig_ext)
  n_dim = len(p_0)
  n_walkers = args.walkers
  # Initialize walkers so that none are out-of-bounds
  if ('l_3' in data_table['type']):
      if args.ms_prior:
          ms = ms_interpolator()
          # Set g3 so that less third star is on the main-sequence at the
          # distance modulus of the binary, 
          g1, Teff1, g2, Teff2, g3, Teff3, ebv = soln.x 
          g_ZAMS_3, g_TAMS_3 = ms(Teff3)
          if (g2 > g1) :
              g_ZAMS_B, g_TAMS_B = ms(Teff2)
              g3 = 0.5*(g_ZAMS_3 + g_TAMS_3) + g2 - 0.5*(g_ZAMS_B+g_TAMS_B)
          else:
              g_ZAMS_B, g_TAMS_B = ms(Teff1)
              g3 = 0.5*(g_ZAMS_3 + g_TAMS_3) + g1 - 0.5*(g_ZAMS_B+g_TAMS_B)
          p_0[4] = g3
      else:
          ms = None
      pos = [p_0]
      for i in range(n_walkers-1):
          lnlike_i = -np.inf
          while lnlike_i == -np.inf:
              pos_i = p_0 + e_p*np.random.randn(n_dim) 
              lnlike_i = lnlike3(pos_i, data_table, ebv_map=ebv_map,
                                 gaussian_ebv=args.gaussian_reddening_prior,
                                 ms_interp=ms)
          pos.append(pos_i)
      sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnlike3,
                args=(data_table, ebv_map, ms), threads=n_threads)

  elif  args.double :
      if args.ms_prior:
          ms = ms_interpolator()
          g1, Teff1, g2, Teff2, ebv = soln.x 
          g_ZAMS_1, g_TAMS_1 = ms(Teff1)
          g_ZAMS_2, g_TAMS_2 = ms(Teff2)
          g2 = 0.5*(g_ZAMS_2 + g_TAMS_2) + g1 - 0.5*(g_ZAMS_1+g_TAMS_1)
          p_0[2] = g2
      else:
          ms = None

      pos = [p_0]

      for i in range(n_walkers-1):
          lnlike_i = -np.inf
          while lnlike_i == -np.inf:
              pos_i = p_0 + e_p*np.random.randn(n_dim) 
              lnlike_i = lnliked(pos_i,data_table,ebv_map=ebv_map,
                      gaussian_ebv=args.gaussian_reddening_prior,
                      ms_interp=ms)
          pos.append(pos_i)
      sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnliked,
              args=(data_table,ebv_map,ms),threads=n_threads)


  elif (lrat > 0) :
      pos = [p_0]
      for i in range(n_walkers-1):
          lnlike_i = -np.inf
          while lnlike_i == -np.inf:
              pos_i = p_0 + e_p*np.random.randn(n_dim) 
              lnlike_i = lnlike2(pos_i,data_table,ebv_map=ebv_map,
                      gaussian_ebv=args.gaussian_reddening_prior)
          pos.append(pos_i)
      sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnlike2,
                args=(data_table,ebv_map),threads=n_threads)
  else:
      pos = [p_0]
      for i in range(n_walkers-1):
          lnlike_i = -np.inf
          while lnlike_i == -np.inf:
              pos_i = p_0 + e_p*np.random.randn(n_dim) 
              lnlike_i = lnlike1(pos_i,data_table,ebv_map=ebv_map,
                      gaussian_ebv=args.gaussian_reddening_prior)
          pos.append(pos_i)
      sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnlike1,
                args=(data_table,ebv_map),threads=n_threads)

    
  meter_width=48
  start_time = time_()

  print('\n Starting emcee chain of {} steps with {} walkers'.
         format(n_steps,n_walkers))

  if args.no_progress_meter:
      sampler.run_mcmc(pos, n_steps)
  else:
      for i, result in enumerate(sampler.sample(pos, iterations=n_steps)):
          n = int((meter_width+1) * float(i) / n_steps)
          delta_t = time_()-start_time 
          time_incr = delta_t/(float(i+1) / n_steps) # seconds per increment
          time_left = time_incr*(1- float(i) / n_steps)
          m, s = divmod(time_left, 60)
          h, m = divmod(m, 60)
          sys.stdout.write("\r[{}{}] {:05.1f}% - {:02.0f}h:{:02.0f}m:{:04.1f}s".
                       format('#' * n, ' ' * (meter_width - n), 
                              100*float(i)/n_steps,h,m,s))

  af = sampler.acceptance_fraction
  print('\n Median acceptance fraction = {:.3f}'.format(np.median(af)))
  best_index = np.unravel_index(np.argmax(sampler.lnprobability),
      (n_walkers, n_steps))
  best_lnlike =  np.max(sampler.lnprobability)
  print(' Best log(likelihood) = {:.2f} in walker {} at step {} '.format(
        best_lnlike,1+best_index[0],1+best_index[1]))

  if ('l_3' in data_table['type']):
      parnames = ["g_1    ", "T_eff,1", "g_2    ", "T_eff,2", 
                  "g_3    ", "T_eff,3", "E(B-V) " , "sig_ext"]
  elif (lrat > 0) | args.double :
      parnames = ["g_1    ", "T_eff,1", "g_2    ", "T_eff,2", "E(B-V) " , 
                  "sig_ext"]
  else:
      parnames = ["g_0    ", "T_eff", "E(B-V) " , "sig_ext"]

  param = sampler.chain[best_index[0],best_index[1],:]
  medians = np.zeros_like(param)
  print ('\n Parameter median values, standard deviations and best-fit values.')
  for i,n in enumerate(param):
      m = np.median(sampler.chain[:,args.burn_in:,i])
      medians[i] = m
      s = np.std(sampler.chain[:,args.burn_in:,i])
      if (m > 1000):
        print("  {} =   {:4.0f} +/- {:4.0f} K     [ {:4.0f} ]".format(
          parnames[i],m, s, param[i]))
      else:
        print("  {} = {:6.3f} +/- {:6.3f}   [ {:6.3f} ]".format(
          parnames[i],m, s, param[i]))
            
  if ('l_3' in data_table['type']):
      print("  chi-squared = {:.2f}".format(
            chisq3(param[:-1], data_table, param[-1])))
      parlabels = ["g$_1$   ", "T$_{eff,1} [K]$", "g$_2$   ", 
                   "T$_{eff,2} [K]$", "g$_3$   ",
                   "T$_{eff,3} [K]$", "E(B-V) " , "$\sigma_{ext}$"]
      results_table = lnlike3(param, data_table,
              gaussian_ebv=args.gaussian_reddening_prior, return_fit=True)
  elif   args.double :
      print("  chi-squared = {:.2f}".format(
            chisqd(param[:-1], data_table, param[-1])))
      parlabels = ["g$_1$   ", "T$_{eff,1} [K]$", "g$_2$-g$_1$ ", 
                   "T$_{eff,2} [K]$", "E(B-V) " , "$\sigma_{ext}$"]
      if args.double:
          results_table = lnliked(param,data_table,
                  gaussian_ebv=args.gaussian_reddening_prior, return_fit=True)
      else:
          results_table = lnlike2(param,data_table,
                  gaussian_ebv=args.gaussian_reddening_prior, return_fit=True)
  elif (lrat > 0) :
      print("  chi-squared = {:.2f}".format(
            chisq2(param[:-1], data_table, param[-1])))
      parlabels = ["g$_1$   ", "T$_{eff,1} [K]$", "g$_2$   ", 
                   "T$_{eff,2} [K]$", "E(B-V) " , "$\sigma_{ext}$"]
      if args.double:
          results_table = lnliked(param,data_table,
                  gaussian_ebv=args.gaussian_reddening_prior, return_fit=True)
      else:
          results_table = lnlike2(param,data_table,
                  gaussian_ebv=args.gaussian_reddening_prior, return_fit=True)
  else:
      print("  chi-squared = {:.2f}".format(
            chisq1(param[:-1], data_table, param[-1])))
      parlabels = ["g$_1$   ", "T$_{eff,1} [K]$", "E(B-V) " , "$\sigma_{ext}$"]
      results_table = lnlike1(param,data_table,
              gaussian_ebv=args.gaussian_reddening_prior, return_fit=True)

  results_table.rename_column("value_1","value_obs")
  results_table.rename_column("value_2","value_fit")
  results_table.pprint(max_lines=-1)
 # Check Teff is within limits of surface brightness calibration 
 # Limits from Table 5 of Graczyk et al. are B-K = [-0.12:3.15]
 # This corresponds to Teff =~ [4915, 9450]
  
  t = Table(sampler.flatchain,names=parnames, masked=True)
  t.add_column(Column(sampler.flatlnprobability,name='loglike'))
  indices = np.mgrid[0:n_walkers,0:n_steps]
  step = 1 + indices[1].flatten() - args.burn_in
  walker = 1 + indices[0].flatten()
  t.add_column(Column(step,name='step'))
  t.add_column(Column(walker,name='walker'))
  t = t[step > 0]
  t.write(args.chain_file,overwrite=args.overwrite)
  print(" Nobs = {}".format(len(data_table)))
  print(" Nmag = {}".format(sum(data_table['type'] == 'mag')))
  print(" Ndf  = {}".format(len(data_table)-len(param)))
  print(' BIC = {:.2f} '.format(
        np.log(len(data_table))*len(param) -2*best_lnlike))
  print("\n")
   
  print("\nCompleted analysis of {}\n".format(args.phot_file))
