from blueqat import Circuit, BlueqatGlobalSetting

# %%
def mark(c, val):
    if val == 0b00:
        return c.s[:].cz[0, 1].s[:]
    elif val == 0b01:
        return c.s[1].cz[0, 1].s[1]
    elif val == 0b10:
        return c.s[0].cz[0, 1].s[0]
    elif val == 0b11:
        return c.cz[0, 1]

def amplify(c):
    return c.h[:].x[:].cz[0, 1].x[:].h[:]

def grover(c, val):
    for _ in range(1): # time of √N (N is qubits)
        c += Circuit().mark(val).amplify()
    return c

# BlueqatGlobalSetting.unregister_macro('mark')
# BlueqatGlobalSetting.unregister_macro('amplify')
# BlueqatGlobalSetting.unregister_macro('grover')
BlueqatGlobalSetting.register_macro('mark', mark)
BlueqatGlobalSetting.register_macro('amplify', amplify)
BlueqatGlobalSetting.register_macro('grover', grover)

# %% practice

result = Circuit(2).h[:].grover(0b01).m[:].run(shots=100)
print(result)
