# PA6, CS124, Stanford, Fall 2021
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
# This submission by Anton Lok, Diego Valdez, Sophie Fujiwara, and Tom Nguyen.
######################################################################
# noinspection PyMethodMayBeStatic

import util
import re
import numpy as np
import random
from porter_stemmer import PorterStemmer


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        self.name = 'HAL'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.stemmer = PorterStemmer()

        self.processed_opinions = 0
        self.user_ratings = np.zeros(len(self.titles))

        self.recommendation_index = 0

        self.neg_msg = ['I am sorry to hear that. It makes my compiler sad. Did I make you feel ', 'Oh no! Please don\'t feel that way, did I make you feel ',
                                'Noooooo! I\'m afraid I can\'t help with that. Was it something I said? Why are you feeling ']
        self.pos_msg = ['? My sincerest apologies. I should not have said that, I can only recommend movies.', '? I am only made to help with movies! Sorry for the confusion.',
                                '? I can suggest talking to a friend. However, I can only help with movies, please tell me about one, it might make you feel better!']
        self.is_happy = ['I\'m so glad that you feel ', 'I can see through my camera that you feel ',
                                'Wow! I love that you feel ']
        self.happy_ending = ['! Let me help you out with a movie.', '! My gut tells me that I should reccomend you a movie, please tell me about one.', '! That makes my compiler feel happy! It would make me more happy if you told me about a movie.']
        self.get_word = re.compile('[^a-zA-Z0-9]')
        self.neg_words = ['mad', 'tired', 'annoyed', 'bitter', 'angry', 'bugged', 'cranky', 'disgusted', 'furious', 'agitated', 'distraught', 'fed up', 'livid', 'fuming',
                           'enraged', 'critical', 'fuming', 'upset', 'awful', 'crushed', 'depressed', 'lonely', 'distressed', 'ashamed', 'desolate',
                           'infuriated', 'irritated', 'anxious', 'awkward', 'appaled', 'fearful', 'frightened', 'desperate', 'fretful', 'afraid', 'alarmed', 'hate', 'hated', 'sad', 'despise', 'despises']

        self.pos_words = ['happy', 'good', 'love', 'amused', 'ecstatic', 'glad', 'joyful', 'proud', 'thrilled', 'ecstatic', 'gleeful'
                           'content', 'excited', 'peaceful', 'satisfied', 'appreciate', 'adore', 'releived', 'awe', 'calm', 'cheerful', 'cool'
                           'elated', 'elevated', 'charmed', 'bold', 'confident', 'amazing', 'great', 'wonderful', 'loved', 'like', 'likes', 'loves']

        self.list_question_words = ['tell', 'have', 'what', 'how', 'what\'s', 'are', 'do', 'if', 'where', 'who', 'tell', 'why', 'will', 'can']
        self.generic_response = ["I can really only help with movies... why don't you tell me about one. I'm happy to help with this!",
                         "Hm, that's not really what I want to talk about right now.", "Looks like a cat ran across the keybord. Why don't you tell me about a movie."
                         "Uhhhhh beep boop I can't really do anything with this. But, can you please tell me about a movie you like or don't like?",
                         "My programmers didn't teach me how to read this. Dang, sorry about that. Tell me about a movie though!!"]
        self.where_response = ['Shh... that\s secret.', 'I can\'t tell you anything about that!', 'Honestly, I don\'t really know...', 'That\'s no fun is it?']

        # Import and stem the sentiment file.
        # Creates a dictionary of stemmed word -> sentiment value.
        self.stemmed_dict = {}
        with open("data/sentiment.txt") as sentiment_file:
            for line in sentiment_file:
                line = line.rstrip()
                (key, val) = line.split(',')
                if val == 'pos':
                    self.stemmed_dict[self.stemmer.stem(key)] = 1
                else:
                    self.stemmed_dict[self.stemmer.stem(key)] = -1

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings, threshold=2.5)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        greeting_message = "Greetings, user. I am HAL. I'm going to recommend a movie to you. First, I need your " \
                           "taste in movies. Please, tell me about a movie that you have seen. "
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        goodbye_message = "Goodbye, now. I will be here next time you need me."
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        if self.creative:
            # If we don't yet have enough information, enter the querying conversation.
            if self.processed_opinions < 5:

                preprocessed_line = self.preprocess(line)
                input_title = self.extract_titles(preprocessed_line)

                #########################
                # STARTS USER INTERACTIONS
                #########################

                # This group intercepts the cases where the sentence does not contain a title + User interaction
                if len(input_title) == 0:
                    response = ''
                    list_words = preprocessed_line.split()
                    confused_str = 'I\'m sorry, I don\'t think I can understand that. Can you clarify or tell me how you feel about a movie?'
                    if len(list_words) < 3:
                        return confused_str

                    flag = 0
                    flag_generic_message = 0
                    for word in list_words:
                        word = self.get_word.sub('', word)
                        if word != '':
                            word = word.lower()
                            if word in self.pos_words or word in self.neg_words:
                                flag = 1
                            elif word in self.list_question_words:
                                flag_generic_message = 1

                    if len(input_title) == 0 and flag == 1 and flag_generic_message == 0:
                        processed_emotion = self.emotion_check(line)

                        neg_response = "Was it something I said? I didn\'t mean that I promise!! But.. please tell me about a movie!"
                        if processed_emotion[1] in ['love', 'loves'] or processed_emotion[1] in ['like', 'likes']:
                            if processed_emotion[2] == True:
                                return neg_response
                            return 'I am flattered and a little confused. Uhhhh, can you please tell me about a movie...'
                        elif processed_emotion[1] in ['hate', 'hates'] or processed_emotion[1] in ['despise',
                                                                                                   'despises']:
                            if processed_emotion[2] == True:
                                return "Oh, ok...."
                            return neg_response

                        if processed_emotion[2] == False:
                            response += processed_emotion[0] + processed_emotion[1]
                        else:
                            return 'Oh ok, did I made you feel not ' + processed_emotion[1] + '? Cool I guess... could you tell me more about a movie?'
                        ending = ''
                        if processed_emotion[1] in self.pos_words:
                            ending = self.happy_ending[random.randint(0, len(self.happy_ending) - 1)]
                        else:
                            ending = self.pos_msg[random.randint(0, len(self.pos_msg) - 1)]

                        return response + ending

                    elif flag_generic_message == 1:
                        for i in range(len(list_words)):
                            if i < 1:
                                list_words[i] = list_words[i].lower()

                        if len(list_words) == 3 and 'why' not in list_words:
                            if 'where' in list_words:
                                if 'you' not in list_words:
                                    return 'I can\'t disclose any information about ' + list_words[-1] + '.'
                                return self.where_response[random.randint(0, len(self.where_response) - 1)]

                            elif 'are' in list_words:
                                if 'how' in list_words or 'what' in list_words or 'who' in list_words:
                                    return 'I do not have permission to tell you anything about myself. Enter the passcocode to find out. Sike, there is no passcode.'
                                return "I can only recommend movies, so sadly I can't answer if I am " + list_words[
                                    -1] + "."

                            elif 'can' in list_words and 'you' in list_words:
                                return "I'm afraid I can't " + list_words[-1] + ". I wasn't programmed to."

                            elif 'how' in list_words:
                                return 'I am just a robot! I do not have any feelings or thoughts... maybe.'

                        elif 'where' in list_words:
                            start = 'Whoops! Looks like I can\'t tell you any information about '
                            if 'the' in list_words:
                                indx = list_words.index('the')
                                if indx + 1 <= len(list_words):
                                    return start + 'the ' + list_words[indx + 1] + '.'
                            indx = list_words.index('is')
                            if indx + 1 <= len(list_words):
                                return start + list_words[indx + 1] + '.'

                        elif 'can' in list_words and 'about' in list_words:
                            start = 'Actually no I can\'t, I know nothing about '
                            indx = list_words.index('about')
                            if 'the' in list_words:
                                indx = list_words.index('the')
                                if indx + 1 <= len(list_words):
                                    return start + 'the' + list_words[indx + 1] + '.'
                            if indx + 1 <= len(list_words):
                                return start + list_words[indx + 1] + '.'

                        elif 'about' in list_words:
                            indx = list_words.index('about')
                            if 'the' in list_words:
                                indx = list_words.index('the')
                                if indx + 1 <= len(list_words):
                                    return 'Whoops! Looks like I have nothing about the' + list_words[indx + 1] + '.'
                            if indx + 1 <= len(list_words):
                                return 'Whoops! Looks like I have nothing about ' + list_words[indx + 1] + '.'

                        elif 'what' in list_words and 'are' not in list_words:
                            return 'Good ol\' HAL does not know anything about ' + list_words[-1] + '.'

                        elif 'how' in list_words and 'you' in list_words:
                            return 'I am just a robot! I do not have any feelings or thoughts... maybe.'
                    else:
                        return confused_str
                    return self.generic_response[random.randint(0, len(self.generic_response) - 1)]

                #########################
                # ENDS USER INTERACTIONS
                #########################
                # In the case where we have multiple titles
                elif len(input_title) >= 2:

                    # Quick check for if the mentioned movies actually exist.
                    potential_titles = self.find_movies_by_title(input_title[0])
                    if len(potential_titles) == 0:
                        response = "Hmm, I don't know of movies by those titles. Tell me about a movie that you have seen."
                        return response
                    sentiment_list = self.extract_sentiment_for_movies(preprocessed_line)
                    response = "You"
                    for movie in sentiment_list:
                        (this_title, this_sentiment) = movie
                        if this_sentiment == 1:
                            response += " liked " + this_title + " and"
                            movie_index = self.find_movies_by_title(this_title)[0]
                            self.user_ratings[movie_index] = 1
                            self.processed_opinions += 1

                        if this_sentiment == -1:
                            response += " did not like " + this_title + " and"
                            movie_index = self.find_movies_by_title(this_title)[0]
                            self.user_ratings[movie_index] = -1
                            self.processed_opinions += 1

                    response = response[:len(response) - 4]
                    response += "."

                    if self.processed_opinions == 5:
                        self.recommendations = self.recommend(self.user_ratings, self.ratings)
                        response += "\n That's enough for me to make a recommendation.\n I recommend you watch " + \
                                    self.titles[self.recommendations[self.recommendation_index]][
                                        0] + ".\n Want another recommendation? Yes or :quit, please."
                        self.recommendation_index += 1
                    return response

                # Having verified that we have a title in the input, we move on to referencing it against the database.
                user_title = input_title[0]
                potential_titles = self.find_movies_by_title(user_title)

                # This group intercepts the cases where there is no match or multiple matches for the provided title.
                if len(potential_titles) == 0:
                    potential_typo_titles = self.find_movies_closest_to_title(user_title, max_distance=3)
                    if len(potential_typo_titles) == 0:
                        response = "Hmm, I don't know of movies by those titles. Tell me about a movie that you have seen."
                        return response
                    else:
                        potential_titles = potential_typo_titles
                if len(potential_titles) >= 2:
                    titles = self.titles[potential_titles[0]][0]
                    for i in range(len(potential_titles) - 1):
                        titles += ', or ' + self.titles[potential_titles[i + 1]][0]
                    clarification = input("That one has been released multiple times. Did you mean " + titles + "?\n")
                    preprocessed_title = self.preprocess(clarification)
                    new_title = self.extract_titles(preprocessed_title)
                    if not new_title:
                        response = "Hmm, I don't which one you're talking about. Tell me about a movie that you have seen."
                        return response
                    potential_titles = self.find_movies_by_title(new_title[0])

                # Having isolated the movie that the user is talking about, we can determine their take on it.
                database_title = self.titles[potential_titles[0]][0]

                if self.extract_sentiment(preprocessed_line) == 1:
                    response = "You liked " + user_title + ". Thank you! Tell me about another movie you have seen."
                    self.user_ratings[potential_titles[0]] = 1
                    self.processed_opinions += 1

                    # To avoid prompting again when five responses have been collected, we need to give the first recommendation right away.
                    if self.processed_opinions == 5:
                        self.recommendations = self.recommend(self.user_ratings, self.ratings)
                        response = "You liked " + user_title + ". Thank you!\n That's enough for me to make a recommendation.\n I recommend you watch " + \
                                   self.titles[self.recommendations[self.recommendation_index]][
                                       0] + ".\n Want another recommendation? Yes or :quit, please."
                        self.recommendation_index += 1

                    return response

                elif self.extract_sentiment(preprocessed_line) == -1:
                    response = "You didn't like " + user_title + ". Thank you! Tell me about another movie you have seen."
                    self.user_ratings[potential_titles[0]] = -1
                    self.processed_opinions += 1

                    # To avoid prompting again when five responses have been collected, we need to give the first recommendation right away.
                    if self.processed_opinions == 5:
                        self.recommendations = self.recommend(self.user_ratings, self.ratings)
                        response = "You didn't like " + user_title + ". Thank you!\n That's enough for me to make a recommendation.\n I recommend you watch " + \
                                   self.titles[self.recommendations[self.recommendation_index]][
                                       0] + ".\n Want another recommendation? Yes or :quit, please."
                        self.recommendation_index += 1

                    return response

                else:
                    response = "I'm sorry, I'm not quite sure if you liked" + user_title + ". Tell me about another movie you have seen."
                    return response

            # We now have five opinions, which is enough to generate a recommendation.
            else:
                recommendations = self.recommend(self.user_ratings, self.ratings)
                if self.recommendation_index == len(recommendations) - 1:
                    response = "My last recommendation is " + self.titles[recommendations[self.recommendation_index]][
                        0] + ". Please enter :quit now!"
                else:
                    response = "I recommend you watch " + self.titles[recommendations[self.recommendation_index]][
                        0] + ".\n Want another recommendation? Yes or :quit, please."
                    self.recommendation_index += 1
                return response

            ########################################################################
            # Standard mode begins here.
            ########################################################################
        else:
            # If we don't yet have enough information, enter the querying conversation.
            if self.processed_opinions < 5:

                preprocessed_line = self.preprocess(line)
                input_title = self.extract_titles(preprocessed_line)

                # This group intercepts the cases where the sentence does not contain a title, or contains multiple titles.
                if len(input_title) == 0:
                    response = "Hmm, I didn't see a title there. Tell me about a movie that you have seen."
                    return response
                elif len(input_title) >= 2:
                    response = "Sorry, I can only handle one at a time. Tell me about a movie that you have seen."
                    return response

                # Having verified that we have a title in the input, we move on to referencing it against the database.
                user_title = input_title[0]
                potential_titles = self.find_movies_by_title(user_title)

                # This group intercepts the cases where there is no match or multiple matches for the provided title.
                if len(potential_titles) == 0:
                    response = "Hmm, I don't know a movie by that title. Tell me about a movie that you have seen."
                    return response
                elif len(potential_titles) >= 2:
                    response = "That one has been released multiple times. Could you tell me about it in the format \"name (year)\" please? "
                    return response

                # Having isolated the movie that the user is talking about, we can determine their take on it.
                database_title = self.titles[potential_titles[0]][0]

                if self.extract_sentiment(preprocessed_line) == 1:
                    response = "You liked " + user_title + ". Thank you! Tell me about another movie you have seen."
                    self.user_ratings[potential_titles[0]] = 1
                    self.processed_opinions += 1

                    # To avoid prompting again when five responses have been collected, we need to give the first recommendation right away.
                    if self.processed_opinions == 5:
                        self.recommendations = self.recommend(self.user_ratings, self.ratings)
                        response = "You liked " + user_title + ". Thank you!\n That's enough for me to make a recommendation.\n I recommend you watch " + \
                                   self.titles[self.recommendations[self.recommendation_index]][
                                       0] + ".\n Want another recommendation? Yes or :quit, please."
                        self.recommendation_index += 1

                    return response

                elif self.extract_sentiment(preprocessed_line) == -1:
                    response = "You didn't like " + user_title + ". Thank you! Tell me about another movie you have seen."
                    self.user_ratings[potential_titles[0]] = -1
                    self.processed_opinions += 1

                    # To avoid prompting again when five responses have been collected, we need to give the first recommendation right away.
                    if self.processed_opinions == 5:
                        self.recommendations = self.recommend(self.user_ratings, self.ratings)
                        response = "You didn't like " + user_title + ". Thank you!\n That's enough for me to make a recommendation.\n I recommend you watch " + \
                                   self.titles[self.recommendations[self.recommendation_index]][
                                       0] + ".\n Want another recommendation? Yes or :quit, please."
                        self.recommendation_index += 1

                    return response

                else:
                    response = "I'm sorry, I'm not quite sure if you liked" + user_title + ". Tell me about another movie you have seen."
                    return response

            # We now have five opinions, which is enough to generate a recommendation.
            else:
                recommendations = self.recommend(self.user_ratings, self.ratings)
                if self.recommendation_index == len(recommendations) - 1:
                    response = "My last recommendation is " + self.titles[recommendations[self.recommendation_index]][
                        0] + ". Please enter :quit now!"
                else:
                    response = "I recommend you watch " + self.titles[recommendations[self.recommendation_index]][
                        0] + ".\n Want another recommendation? Yes or :quit, please."
                    self.recommendation_index += 1
                return response

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def word_negate(self, word_prev, back_word):
        if word_prev in ['not', 'never'] or word_prev.endswith('nt'):
            return True
        if back_word in ['not', 'never'] or back_word.endswith('nt'):
            return True
        return False

    def emotion_check(self, input):
        list_words = input.split()
        pos_msg = self.is_happy[random.randint(0, len(self.is_happy) - 1)]
        neg_msg = self.neg_msg[random.randint(0, len(self.neg_msg) - 1)]

        word_prev = ''
        back_word = ''
        word_in_line = ''
        for word in list_words:
            word = self.get_word.sub('', word)
            if word != '':
                word = word.lower()
                if word in self.pos_words:
                    word_in_line = word
                    if self.word_negate(word_prev, back_word):
                        return neg_msg, word_in_line, True
                    return pos_msg, word_in_line, False
                if word in self.neg_words:
                    word_in_line = word
                    if self.word_negate(word_prev, back_word):
                        return pos_msg, word_in_line, True
                    return neg_msg, word_in_line, False
            back_word = word_prev
            word_prev = word

        return '', word_in_line, False

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        # Creates a dictionary of contraction -> expanded form. Adapted from:
        # https: // en.wikipedia.org / wiki / Wikipedia: List_of_English_contractions
        contractions = {}
        with open("deps/contractions.txt") as contractions_file:
            for line in contractions_file:
                line = line.rstrip()
                (key, val) = line.split(',')
                contractions[key] = val

        for contraction in contractions:
            text = text.replace(contraction, contractions[contraction])

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """

        title_pattern = '\"(.+?)\"'
        title_matches = re.findall(title_pattern, preprocessed_input)
        return title_matches

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        matches = []
        if len(title) >= 8 and title[-6] == '(':
            has_year = True
            name = title[:-7]
            year = title[-6:]
        else:
            has_year = False
            name = title
        new_name = ""

        if name[0:2] == "A ":
            new_name = name[2:] + ", A"

        if name[0:3] == "An ":
            new_name = name[3:] + ", An"

        if name[0:4] == "The ":
            new_name = name[4:] + ", The"

        if self.creative:
            for index in range(len(self.titles)):
                found = re.search(r'\b{0}\b'.format(name), self.titles[index][0])
                foreign = re.search(r'\(' + name + '\)'.format(name), self.titles[index][0])
                foreign_1 = re.search(r'\(a.k.a.' + name + '\)'.format(name), self.titles[index][0])
                if found or foreign or foreign_1:
                    matches.append(index)
                if new_name != "" and new_name == self.titles[index][0][:-7]:
                    matches.append(index)
        else:
            for index in range(len(self.titles)):
                if name == self.titles[index][0][:-7]:
                    matches.append(index)
                if new_name != "" and new_name == self.titles[index][0][:-7]:
                    matches.append(index)

        final_matches = []
        for title_index in matches:
            if has_year and self.titles[title_index][0][-6:] == year:
                final_matches.append(title_index)

        if has_year:
            return final_matches

        return matches

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        # Remove titles, punctuation, and split into word tokens.
        titles_removed = re.sub('\s?".*?"\s?', '', preprocessed_input)
        punctuation_removed = re.sub('[^A-Za-z0-9 ]+', '', titles_removed)
        list_of_words = punctuation_removed.split(' ')

        # Stem the words from the input.
        stemmed_words = []
        for word in list_of_words:
            stemmed_words.append(self.stemmer.stem(word))

        # Additional processing to identify negation.
        negation_word_count = 0
        # TODO : Add more negation words if appropriate.
        negation_words = ['cannot', 'never', 'not', 'no']
        for word in negation_words:
            if word in stemmed_words:
                negation_word_count += 1

        # Determine the raw score using the processed tokens.
        current_score = 0
        for word in stemmed_words:
            if word in self.stemmed_dict:
                current_score += self.stemmed_dict[word]

        # Synthesize the score and negation count, and return the result.
        if current_score > 0:
            if negation_word_count % 2 == 0:
                return 1
            else:
                return -1
        elif current_score < 0:
            if negation_word_count % 2 == 0:
                return -1
            else:
                return 1
        else:
            return 0

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        sentence_splits = preprocessed_input.split("\"")
        movies = self.extract_titles(preprocessed_input)
        new_sentence_splits = []
        negation_words = ['but', 'never', 'not', 'no']
        for index in range(len(sentence_splits)):
            if sentence_splits[index] in movies:
                movie_with_quotes = "\"" + sentence_splits[index] + "\""
                new_fragment = sentence_splits[index - 1] + movie_with_quotes
                new_sentence_splits.append(new_fragment)
        movie_sentiments = []
        negation = False
        for index in range(len(new_sentence_splits)):
            sentiment = self.extract_sentiment(new_sentence_splits[index])
            if sentiment == 0:
                found = False
                for word in negation_words:
                    found = re.search(r'\b{0}\b'.format(word), new_sentence_splits[index])
                    if found:
                        break
                if found:
                    movie_sentiments.append(movie_sentiments[index - 1] * -1)
                else:
                    movie_sentiments.append(movie_sentiments[index - 1])
            else:
                movie_sentiments.append(sentiment)
        output = []
        for i in range(len(movie_sentiments)):
            output.append((movies[i], movie_sentiments[i]))
        return output

    def minimum_edit_distance(self, input, movie):
        user_movie = " " + input
        movie_title = " " + movie
        array = np.ndarray([len(user_movie), len(movie_title)])
        # matrix is n by m
        # shape(n,m)
        # print(array.shape)
        for num in range(array.shape[0]):
            array[num][0] = num
        for num in range(array.shape[1]):
            array[0][num] = num
        for i in range(1, array.shape[0]):
            for j in range(1, array.shape[1]):
                possible_values = []
                # the left
                possible_values.append(array[i - 1][j] + 1)
                # the right
                possible_values.append(array[i][j - 1] + 1)
                # the diagonal
                if (user_movie[i] == movie_title[j]):
                    possible_values.append(array[i - 1][j - 1])
                else:
                    possible_values.append(array[i - 1][j - 1] + 2)
                minimum = possible_values[0]
                for index in range(len(possible_values)):
                    if (possible_values[index] < minimum):
                        minimum = possible_values[index]
                array[i][j] = minimum
        return array[array.shape[0] - 1][array.shape[1] - 1]

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        distances = []
        for name in self.titles:
            new_name = name

            if re.search(", The", name[0]):
                new_name = "The " + re.sub(', The', '', name[0])
            if re.search(", An", name[0]):
                new_name = "An " + re.sub(', An', '', name[0])
            if re.search(", A", name[0]):
                new_name = "A " + re.sub(', A', '', name[0])

            distances.append(self.minimum_edit_distance(title.lower(), name[0][:-7].lower()))
        movies_in_range = []
        minimum = max_distance
        for index in range(len(distances)):
            if distances[index] == minimum:
                movies_in_range.append(index)
            elif distances[index] < minimum:
                minimum = distances[index]
                movies_in_range = []
                movies_in_range.append(index)

        return movies_in_range

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        matches = []
        for index in candidates:
            found = re.search(r'\b{0}\b'.format(clarification), self.titles[index][0])
            if found:
                matches.append(index)
            # if new_name != "" and new_name == self.titles[index][0][:-7]:
            #     matches.append(index)
        return matches

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """

        # Make a copy of the matrix.
        binary_ratings = np.copy(ratings)

        # Set low scores to -1.
        binary_ratings[binary_ratings <= threshold] = -1

        # Set high scores to 1.
        binary_ratings[binary_ratings > threshold] = 1

        # Set the zeroes back to zeroes (previously set to low).
        binary_ratings[ratings == 0] = 0

        return binary_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom == 0:
            return 0
        return np.dot(u, v) / denom

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        similarities = {}
        recommendations = []
        rated_movies = []
        for i in range(len(user_ratings)):
            if user_ratings[i] != 0:
                rated_movies.append(i)

        for i in range(len(ratings_matrix)):
            sums = 0
            for j in range(len(rated_movies)):
                sums += self.similarity(ratings_matrix[i], ratings_matrix[rated_movies[j]]) * user_ratings[
                    rated_movies[j]]
            similarities[i] = sums

        new_dict = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        count = 0
        for key, value in new_dict:
            if count == k:
                break
            if user_ratings[key] == 0:
                count += 1
                recommendations.append(key)
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
            This is HAL, a movie recommendation chatbot.
            HAL was designed and implemented by Anton Lok, Diego Valdez, Sophie Fujiwara, and Tom Nguyen.
            To answer HAL's questions, type in your response and press ENTER.
            For best results, answer the exact question that HAL asks.
            Input :quit at any time to shut HAL down.
            """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')