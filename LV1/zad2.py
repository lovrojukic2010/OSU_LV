import sys
try:
    number = float(input("Unesi broj između 0 i 1: "))
except ValueError:
    print("Unesena vrijednost nije broj")
    sys.exit(1)
    
if number < 0.0 or number > 1.0:
    print("Broj nije izmedu 0 i 1")
elif number>=0.9:
    print("Ocjena A")
elif number>=0.8:
    print("Ocjena B")
elif number>=0.7:
    print("Ocjena C")
elif number>=0.6:
    print("Ocjena D")
else:
    print("Ocjena F")
 