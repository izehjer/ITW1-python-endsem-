#Author: Mohmmad Anas Khan 
#Roll no: 20075054
#python end sem q3


import pandas as pd

a = pd.DataFrame( [['Chicken Strips' , '$3.50'],
 ['French Fries' ,'$2.50' ],
 [ 'Hamburger' , '$4.00'],
 ['Hotdog' , '$3.50'],
 ['Large Drink' , '$1.75'],
 ['Medium Drink',  '$1.50'],
 ['Milk Shake' , '$2.25'],
 ['Salad', '$3.75'],
 ['Small Drink' ,'$1.25']],
 index = [1,2,3,4,5,6,7,8,9],
 columns=['food' , 'cost']
)

cost = [3.50 ,2.50,4.00,3.50,1.75,1.50,2.25,3.75,1.25]

while(1):
    v = input('Would you like to give your order:(Y/N) ')
    if v=='Y':
        print(a)
        st=input('Enter your order:(1-9) ')
        total =0 
        for i in range(0,len(st)):
            try:
                total+=cost[int(st[i])-1]
            except:
                print('Invalid entry')
                break
        
        print('Total cost is : ' , total)
            
    elif v=='N':
        print('Thank you for visting')
        break
    else:
        print('Invalid entry')
        
        
