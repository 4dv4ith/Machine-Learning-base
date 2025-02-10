'''Given is a list of of words, wordlist, and a string, name. 
Write a Python function which takes wordlist and name as input and returns a tuple. 
The first element of the output tuple is the number of words in the wordlist which have name as a
substring in it. 
The second element of the tuple is a list showing 
the index at which the name occurs in each of the words of the wordlist and a 0 if it doesn’t occur.'''

def input_wordlist():
    wordlist = []
    number = int(input("Enter the number of elements in the list : "))
    for a in range(number):
        word = input("Enter the word :")
        wordlist.append(word)
    return wordlist
def check_string(name,list1):
    list2 = []
    wordlist = list1
    word = name
    temp =0
    for r,g in enumerate(wordlist,start =1):
        if word in g:
            temp = temp+1
            
            list2.append(r) 
        else :
            list2.append(0)
    return (temp,list2)      
list1 = input_wordlist()
name = input("Enter the word for searching : ")
tuple1 = check_string(name,list1)
print(f"Entered list {list1}")
print(f"tuple :{tuple1}")
