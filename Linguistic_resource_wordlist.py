__author__ = 'mirko'

class WORDLIST(object):


    def __init__(self):

        domainwordlistfile = 'resources/domainwordlist.txt'
        content = None
        with open(domainwordlistfile) as f:
            content = f.readlines()
        domainwordlist = [word.rstrip('\n') for word in content]

        #http://www-personal.umich.edu/~jlawler/wordlist
        wordlistfile = 'resources/wordlist.txt'
        content = None
        with open(wordlistfile) as f:
            content = f.readlines()
        self.wordlist = [word.rstrip('\n') for word in content]

        self.wordlist +=domainwordlist

        return

#http://stackoverflow.com/questions/20516100/term-split-by-hashtag-of-multiple-words
    def ParseSentence(self,sentence):
        new_sentence = "" # output
        terms = sentence.lower().split(' ')
        for term in terms:
            if len(term)<1: # double spaces in the text
                new_sentence += ""
            elif term[0] == '#': # this is hashtag, parse it
                new_sentence += self.ParseTag(term)
            elif term[0] == '@': # this is screnname, parse it
                new_sentence += self.ParseTag(term)
            else: # Just append the word
                new_sentence += term
            new_sentence += " "

        return new_sentence




    def ParseTag(self, term):
        words = []
        # Remove hashtag, split by dash
        tags = term[1:].split('-')
        for tag in tags:
            word = self.FindWord(tag)
            while word != None and len(tag) > 0:
                words += [word]
                if len(tag) == len(word): # Special case for when eating rest of word
                    break
                tag = tag[len(word):]
                word = self.FindWord(tag)
        return " ".join(words)


    def FindWord(self,token):
        i = len(token) + 1
        while i > 1:
            i -= 1
            if token[:i] in self.wordlist:
                return token[:i]
        return None

if __name__ == '__main__':

    sentence="@tedcruz And, #nothill #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #whyImnotvotingHillaryclinton #SemST"
    wordlist=WORDLIST()

    result=wordlist.ParseSentence(sentence)
    print(result)
