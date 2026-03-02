def total_euro(workHours,hourPrice):
    return workHours*hourPrice

workHours = float(input("Unesi broj radnih sati: "))
hourPrice = float(input("Unesi placu po radnom satu: "))
print(f"{total_euro(workHours,hourPrice)} eura")