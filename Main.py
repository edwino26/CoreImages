import glob
import lasio
import matplotlib.pyplot as plt
import images

files = glob.glob('./*.las')
las = lasio.read(files[0])

well = las.well
headers = las.curves
params = las.params
logs = las.data
No_logs = len(headers)
dims = las.data.shape

print(dims)
for curve in las.curves:
    print(curve.mnemonic + ": " + str(curve.data))

print((las.curves['DEPT']))

DEPTH = las.index
GR = las["GR_EDTC"]
RESD = las["AT90"]
RHOB = las["RHOZ"]
NPHI= las["NPHI"]

BD = 3000
TD = 3880


plt.figure()
plt.subplot(141)
plt.plot(GR, DEPTH, 'green'); plt.axis([0, 120, BD, TD]); plt.gca().invert_yaxis()
plt.subplot(142)
plt.plot(RESD, DEPTH); plt.axis([0.1, 100, BD, TD]); plt.gca().invert_yaxis();plt.gca().yaxis.set_visible(False); plt.grid(True,which='minor',axis='x'); plt.xscale('log')
plt.subplot(143)
plt.plot(RHOB, DEPTH, 'red'); plt.axis([1.65, 2.65, BD, TD]); plt.gca().invert_yaxis(); plt.gca().yaxis.set_visible(False)
plt.subplot(144)
plt.plot(NPHI, DEPTH, 'blue')
plt.gca().invert_yaxis(); plt.axis([0.6, 0, BD, TD]); plt.gca().invert_yaxis();plt.gca().yaxis.set_visible(False)

plt.suptitle('Well logs for ' + las.well['WELL']['value'])



plt.show()