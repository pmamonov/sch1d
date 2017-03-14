from numpy import zeros,exp,inf,dot,argsort,linspace,sort,array,arange,sqrt,tile
from numpy.linalg import eig,inv
from scipy.integrate import quad


def genK(x, V):
  h = x[1]-x[0]
  K = zeros((len(x),len(x)))
  for i in xrange(len(x)):
    K[i,i] = quad(lambda r,h=h: ((r-x[i]+h)/h)**2 * V(r), x[i]-h, x[i])[0] + quad(lambda r,h=h: (1-(r-x[i])/h)**2 * V(r), x[i], x[i]+h)[0]
    if i>0:
      K[i,i-1] = quad(lambda r,h=h: (r-x[i-1])/h * (1-(r-x[i-1])/h) * V(r) , x[i-1],x[i])[0]
    if (i<len(x)-1):
      K[i,i+1] = quad(lambda r,h=h: (r-x[i])/h * (1-(r-x[i])/h) * V(r) , x[i],x[i+1])[0]
  return K
      
def genT(x):
  h = x[1]-x[0]
  T = zeros((len(x),len(x)))
  for i in xrange(len(x)):
    T[i,i] = -2./h
    if i>0:
      T[i,i-1] = 1./h
    if (i<len(x)-1):
      T[i,i+1] = 1./h
  return T/2.

def solve(x,V,m=1):
  h=x[1]-x[0]
  T,K = genT(x), genK(x,V)
  e,v = eig(K-T/m)
  v = v / sqrt((v**2).sum(0) * h)

#  e = (v*dot(K-T/m,v)).sum(0)
  d2 = (v[:-2,:] + v[2:,:] - 2*v[1:-1,:])/h**2
  e = h*((-v[1:-1,:]*d2/m/2 + v[1:-1,:]**2 * tile(V(x[1:-1]),(len(x),1)).T )).sum(0)
  ei = argsort(e)
  return e[ei],v[:,ei]

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc('font', size=10)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.colors import rgb2hex,colorConverter

from Tkinter import *
import tkMessageBox, tkSimpleDialog, tkFont
from tkFileDialog import askopenfilename, asksaveasfilename


class SchGUI:
  def __init__(self, master):
    master.wm_title("1D Schrodinger equation in arbitrary potential")
    fnt=tkFont.Font(font=("Helvetica",12,NORMAL))
    master.option_add("*Font", fnt)

    # Figure
    fr_fig = Frame(master,padx=5, pady=5)
    fr_fig.pack(side=LEFT)

    self.f = Figure(figsize=(7,6), dpi=100)
    canvas = FigureCanvasTkAgg(self.f, master=fr_fig)
    canvas.show()
    # Figure toolbar
    toolbar = NavigationToolbar2TkAgg(canvas, fr_fig)
    toolbar.update()
    canvas.get_tk_widget().pack(side=LEFT)
    self.ax = self.f.add_subplot(111)

    fr_cont = Frame(master, padx=5, pady=5)
    fr_cont.pack(side=LEFT, anchor='nw')

    fr_v = Frame(fr_cont, padx=5, pady=5)
    fr_v.pack(side=TOP, anchor='nw')
    Label(fr_v, text="V(x) = ").pack(side=LEFT)
    self.en_v = Entry(fr_v, width=15)
    self.en_v.pack(side=LEFT)
    bt_vplot = Button(fr_v, text="Plot", command=self.plot_v)
    bt_vplot.pack(side=LEFT)

    fr_m = Frame(fr_cont, padx=5, pady=5)
    fr_m.pack(side=TOP, anchor='nw')
    Label(fr_m, text="m = ").pack(side=LEFT)
    self.en_m = Entry(fr_m, width=5)
    self.en_m.pack(side=LEFT)

    fr_x = Frame(fr_cont, padx=5, pady=5)
    fr_x.pack(side=TOP, anchor='nw')
    Label(fr_x, text="x = ").pack(side=LEFT)
    self.en_x1 = Entry(fr_x, width=5)
    self.en_x1.pack(side=LEFT)
    Label(fr_x, text="..").pack(side=LEFT)
    self.en_x2 = Entry(fr_x, width=5)
    self.en_x2.pack(side=LEFT)
    Label(fr_x, text="# points").pack(side=LEFT)
    self.en_np = Entry(fr_x, width=5)
    self.en_np.pack(side=LEFT)

    bt_solve = Button(fr_cont, text="SOLVE", command=self.solve)
    bt_solve.pack(side=TOP)

    fr_plt = Frame(fr_cont, padx=5, pady=5)
    fr_plt.pack(side=TOP, anchor='nw')
    Label(fr_plt, text="WF # ").pack(side=LEFT)
    self.en_wf = Entry(fr_plt, width=5)
    self.en_wf.pack(side=LEFT)
    bt_solve = Button(fr_plt, text="Plot", command=self.plot_wf)
    bt_solve.pack(side=LEFT)

    bt_exp = Button(fr_plt, text="EXPORT", command=self.export)
    bt_exp.pack(side=LEFT)

    bt_clr = Button(fr_cont, text="CLEAR", command=self.clear)
    bt_clr.pack(side=TOP)

    self.clear()

  
  def plot(self, x,y, label=''):
    self.ax.plot(x,y,label=label)
    self.ax.legend()
    self.ax.figure.canvas.draw()

  def clear(self,):
    self.ax.cla()
    self.ax.grid()
    self.ax.figure.canvas.draw()

  def get_x(self):
    self.x = linspace(float(self.en_x1.get()), float(self.en_x2.get()), int(self.en_np.get()))

  def get_v(self):
     self.V = eval("lambda x: %s"%self.en_v.get())
   
  def plot_v(self):
    self.get_x()
    self.get_v()
    self.plot(self.x, self.V(self.x), label=self.en_v.get())

  def solve(self):
    self.get_x()
    self.get_v()
    self.e,self.v = solve(self.x, self.V, float(self.en_m.get()))

  def plot_wf(self):
    wfs = map(int, self.en_wf.get().split('-'))
    if len(wfs) >= 2: wf1,wf2 = wfs[0],wfs[1]+1
    else:  wf1,wf2=wfs[0], wfs[0]+1
    for i in xrange(wf1,wf2):
      self.plot(self.x, self.e[i]+self.v[:,i], label='%.3e'%(self.e[i]))

  def export(self):
    fn = asksaveasfilename()
    if not fn: return
    wfs = map(int, self.en_wf.get().split('-'))
    if len(wfs) >= 2: wf1,wf2 = wfs[0],wfs[1]+1
    else:  wf1,wf2=wfs[0], wfs[0]+1
    f = open(fn, 'w')
    print >>f, '0',
    for i in xrange(wf1,wf2): print >>f, " %e"%self.e[i],
    for i in xrange(self.v.shape[0]):
      print >>f, ""
      print >>f, "%e"%self.x[i],
      for n in xrange(wf1,wf2): print >>f, " %e"%self.v[i,n],
    f.close()


if __name__ == "__main__":
  root = Tk()
  app = SchGUI(root)
  root.mainloop()
