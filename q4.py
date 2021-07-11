#Author: Mohmmad Anas Khan 
#Roll no: 20075054
#python end sem q4


import pickle
import pandas as pd
import os

f = pd.DataFrame([['Add Contact'] , ['Display contact'] , ['Delte Contact'] ,
              ['Modify Contact'],['Search Contatct']],
              index=[1,2,3,4,5],
              columns=['options']
             )

print(f)

v=int(input('Enter an option:(1-5): '))
if v==1:
    name=input('Enter contact name: ')
    mail=input('Enter contacts email-id: ')
    ph=input('Enter contacts phone: ')
    data={}
    data[name]=(mail,ph)
    f=open('/home/mak/contact.txt' , 'wb')
    pickle.dump(data, f)
    f.close()
    print('Contact has been added ')
    
elif v==2:
    f=open('/home/mak/contact.txt','rb')
    d1 = pickle.load(f)
    f.close()
    c=bool(d1)
    if c:
        f=open('/home/mak/contact.txt','rb')
        d = pickle.load(f)
        f.close()
        print(d)
    else:
        print('No contact in address book')
 
elif v==3:
    g = input('enter the name of the contact to be deleted: ')
    f=open('/home/mak/contact.txt','rb')
    d = pickle.load(f)
    f.close() 
    if g in d:
        del d[g]
        f=open('/home/mak/contact.txt' , 'wb')
        pickle.dump(d, f)
        f.close()
        print('contact succesfully removed')
    else:
        print('no contact with this name')
        
    
elif v==4:
    f=open('/home/mak/contact.txt','rb')
    d1 = pickle.load(f)
    f.close()
    c=bool(d1)
    if c:
        g=input('Enter the name of the contact to be modified: ')
    
        f=open('/home/mak/contact.txt','rb')
        d = pickle.load(f)
        f.close()
        if g in d:
            m = input('Enter the new mail id: ')
            j = input('Enter the new phone number: ')
            d[g]=(m,j)
            f=open('/home/mak/contact.txt','wb')
            pickle.dump(d,f)
            f.close()
            print('Contact has been successfully updated: ')
        else:
            print('No contact with this name')
         
    else:
        print('Address book empty. No contact to delete')
   
elif v==5:
    g = input('enter the name of the contact to be searched: ')
    f=open('/home/mak/contact.txt','rb')
    d = pickle.load(f)
    f.close()
    if g in d:
        print(d[g])
    
else:
    print('Invalid option ')