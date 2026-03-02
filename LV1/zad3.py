numbers = []
while True:
    number = input("Unesi broj: ")
    if number == "Done":
        break
    try:
        number = float(number)
    except:
        print("Unesena vrijednost nije broj niti Done")
        continue
    numbers.append(number)
    

if len(numbers)==0:
    print("Lista je prazna")
else:
    numbers = sorted(numbers)
    print(f"Uneseno brojeva: {len(numbers)}")
    print(f"Sortirana lista: {numbers}")
    print(f"Minimalna vrijednost u listi: {min(numbers)}")
    print(f"Maksimalna vrijednost u listi: {max(numbers)}")
    print(f"Srednja vrijednost: {sum(numbers)/len(numbers)}")

