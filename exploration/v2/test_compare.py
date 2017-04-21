
import RAKE
import tfidf
import textrank
import os
import time
import string

RAKE_STOPLIST = 'stoplists/SmartStoplist.txt'
def execute(pages):
    """Execute RAKE and TF-IDF algorithms on each page and output top scoring phrases"""

    start_time = time.time()

    

    #2: Collect raw text for pages
    print("=== 2. Collect Raw Text from file")
    text = ""
    f = open(pages[0], "r")
    for line in f:
        #line = line.strip("\r")
        #line = line.strip("\n") 
        text += line.lower()
    processed_pages = []
    for page in pages:
        page_text = text
        processed_pages.append({"url": pages[0], "text": page_text})
    print("Collected: %d" % (time.time() - start_time))

    #3: RAKE keywords for each page
    print("=== 3. RAKE")
    rake = RAKE.Rake(RAKE_STOPLIST, min_char_length=2, max_words_length=1)
    for page in processed_pages:
        page["rake_results"] = rake.run(page["text"])
    print("RAKE: %d" % (time.time() - start_time))

    #4: TF-IDF keywords for processed text
    print("=== 4. TF-IDF")
    document_frequencies = {}
    document_count = len(processed_pages)
    for page in processed_pages:
        page["tfidf_frequencies"] = tfidf.get_word_frequencies(page["text"])
        for word in page["tfidf_frequencies"]:
            document_frequencies.setdefault(word, 0)
            document_frequencies[word] += 1

    sortby = lambda x: x[1]["score"]
    for page in processed_pages:
        for word in page["tfidf_frequencies"].items():
            word_frequency = word[1]["frequency"]
            docs_with_word = document_frequencies[word[0]]
            word[1]["score"] = tfidf.calculate(word_frequency, document_count, docs_with_word)

        page["tfidf_results"] = sorted(page["tfidf_frequencies"].items(), key=sortby, reverse=True)
    print("TF-IDF: %d" % (time.time() - start_time))

    #5. TextRank
    print("=== 5. TextRank")
    for page in processed_pages:
        textrank_results = textrank.extractKeyphrases(page["text"])
        page["textrank_results"] = sorted(textrank_results.items(), key=lambda x: x[1], reverse=True)
    print("TextRank: %d" % (time.time() - start_time))

    #6. Results
    print("=== 6. Results")
    for page in processed_pages:
        print("-------------------------")
        print("URL: %s" % page["url"])
        print("RAKE:")
        for result in page["rake_results"][:5]:
            print(" * %s" % result[0], result[1])
        print("TF-IDF:")
        for result in page["tfidf_results"][:5]:
            print(" * %s" % result[0], result[1])
        print("TextRank:")
        for result in page["textrank_results"][:5]:
            print(" * %s" % result[0], result[1])
         
    end_time = time.time() - start_time
    print('Done. Elapsed: %d' % end_time)

def main():

    execute(["test.txt"])



main()