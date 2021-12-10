

#input
#IMPORTANT: STARTS AT 0!!!!!!
s_diagonal = [10.2,6.1,2.8,2.2,1.6]
components = [0,4]

s_top,s_bot = 0,0

for i in components:
    s_top +=s_diagonal[i]*s_diagonal[i]
    
for j in s_diagonal:
    s_bot += j*j
    
explained = s_top/s_bot
if(components[0]==1):
    print("STARTS AT 0 ASSHOLE!")
    print("STARTS AT 0 ASSHOLE!")
    print("STARTS AT 0 ASSHOLE!")
    print("STARTS AT 0 ASSHOLE!")

print(explained)