#TODO mettre le bon nom de fichier
data = np.load('') 
Utr = data['Utr']  
Ytr = data['Ytr']  
Uts1 = data['Uts1'] 
Uts2 = data['Uts2'] 
print(f"data loaded: Utr {Utr.shape}, Ytr {Ytr.shape}")