from apiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

from DataProcessing.Clustering.KMeansClustering import KMeansClusterer
from DataProcessing.QuestionAnswering.DistilbertQuestionAnswering import DistilbertQuestionAnswering
from DataProcessing.SentenceCleanser import SentenceCleanser
# from DataProcessing.TextSummarization.Abstractive.BARTAbstractiveSummarizer import BARTAbstractiveSummarizer
# from DataProcessing.TextSummarization.Abstractive.T5SmallSummarizer import T5SmallSummarizer
from DataProcessing.TextSummarization.Extractive.SumyLexRankSummarizer import SumyLexRankSummarizer
from DataProcessing.LanguageDetectorMain.LanguageDetectorMain import LanguageDetectorMain
from Security.OAuth2Security import get_settings
# from Security.OAuth2Security import OAuth2Security
from DataProcessing.SentenceDetectionGeneratorDetector import SentenceTypeDetection
from DataProcessing import WrapText
from DataProcessing.SentenceDetectionGeneratorDetector.SentenceTypeDetectorPOS import SentenceTypeDetectorPOS

import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

# DEVELOPER_KEY = ""
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


# class OAuth2Security:
#     pass


class YoutubeSearch:

    def __init__(self):
        self.language_processing_model = LanguageDetectorMain()
        self.sentence_cleanser = SentenceCleanser()
        # self.text_summarizer = TextSummarization.TextSummarizer()
        self.lstm_load = SentenceTypeDetectorPOS()
        self.summarizer_model_map = self.initialize_summarizer_models()
        self.ques_ans_model_map = self.initialize_question_answering_models()
        self.kmeansClusterer = KMeansClusterer()
        # self.oAuth2Security = OAuth2Security()
        self.youtube_object = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                               developerKey=get_settings().YOUTUBE_DEVELOPER_KEY)

    def initialize_summarizer_models(self):
        models = {
            # "Extractive - BertExtractiveSummarizer": BertExtractiveSummarizer(),
            # "Extractive - NLTKSummarizer": NLTKSummarizer(),
            "Extractive - SumyLexRankSummarizer": SumyLexRankSummarizer(),
            # "Extractive - SumyLSARankSummarizer": SumyLSARankSummarizer(),
            # "Extractive - SumyLuhnSummarizer": SumyLuhnSummarizer(),
            # "Extractive - SumyTextRankSummarizer": SumyTextRankSummarizer(),

            # "Abstractive - BARTAbstractiveSummarizer": BARTAbstractiveSummarizer(),
            # "Abstractive - DistilbertSummarizer": DistilbertSummarizer(),
            # "Abstractive - T5BaseSummarizer": T5BaseSummarizer(),
            # "Abstractive - T5SmallSummarizer": T5SmallSummarizer()
        }
        return models

    def initialize_question_answering_models(self):
        models = {
            "DistilbertQuestionAnswering": DistilbertQuestionAnswering()
        }
        return models

    def get_summarizer_model_list(self):
        return sorted(list(self.summarizer_model_map.keys()))

    def get_question_answering_model_list(self):
        return sorted(list(self.ques_ans_model_map.keys()))

    def youtube_get_videos(self, query, max_results):
        search_keyword = self.youtube_object.search().list(q = query, part = "id, snippet",
                                                   maxResults = max_results).execute()
        results = search_keyword.get("items", [])
        videos = []
        playlists = []
        channels = []
        for result in results:
            if result['id']['kind'] == "youtube#video":
                videos.append({"title" :result["snippet"]["title"],
                                "videoId":result["id"]["videoId"], "description" :result['snippet']['description'],
                                "thumbnails" :result['snippet']['thumbnails']['default']['url']})
            # elif result['id']['kind'] == "youtube#playlist":
            #     playlists.append("% s (% s) (% s) (% s)" % (result["snippet"]["title"],
            #                          result["id"]["playlistId"],
            #                          result['snippet']['description'],
            #                          result['snippet']['thumbnails']['default']['url']))
            #
            # elif result['id']['kind'] == "youtube#channel":
            #     channels.append("% s (% s) (% s) (% s)" % (result["snippet"]["title"],
            #                            result["id"]["channelId"],
            #                            result['snippet']['description'],
            #                            result['snippet']['thumbnails']['default']['url']))

        return {"videos": videos,
                "itemCount": search_keyword['pageInfo'],
                "nextPageToken": search_keyword["nextPageToken"] if ("nextPageToken" in search_keyword) else None,
                "prevPageToken": search_keyword["prevPageToken"] if ("prevPageToken" in search_keyword) else None}

    def youtube_get_videos_by_token(self, searchText, page_token, max_results):
        print("Calling youtube_get_videos_by_token()")
        print("token - ", page_token)
        print("max_results - ", max_results)
        search_keyword = self.youtube_object.search().list(q = searchText, pageToken = page_token, part = "id, snippet",
                                                   maxResults = max_results).execute()
        results = search_keyword.get("items", [])
        videos = []
        playlists = []
        channels = []
        for result in results:
            if result['id']['kind'] == "youtube#video":
                videos.append({"title" :result["snippet"]["title"],
                                "videoId":result["id"]["videoId"], "description" :result['snippet']['description'],
                                "thumbnails" :result['snippet']['thumbnails']['default']['url']})
            # elif result['id']['kind'] == "youtube#playlist":
            #     playlists.append("% s (% s) (% s) (% s)" % (result["snippet"]["title"],
            #                          result["id"]["playlistId"],
            #                          result['snippet']['description'],
            #                          result['snippet']['thumbnails']['default']['url']))
            #
            # elif result['id']['kind'] == "youtube#channel":
            #     channels.append("% s (% s) (% s) (% s)" % (result["snippet"]["title"],
            #                            result["id"]["channelId"],
            #                            result['snippet']['description'],
            #                            result['snippet']['thumbnails']['default']['url']))

        return {"videos": videos,
                "itemCount": search_keyword['pageInfo'],
                "nextPageToken": search_keyword["nextPageToken"] if ("nextPageToken" in search_keyword) else None,
                "prevPageToken": search_keyword["prevPageToken"] if ("prevPageToken" in search_keyword) else None}


    '''
    Extracts out youtube comments, description and possible question and answer for a given video ID.
    '''
    def youtube_get_comments(self, video_id, max_results, statements, questions, classifier, max_results_replies):

        comments = []
        try:
            # Fetchin all the comments
            comment_objects = self.youtube_object.commentThreads().list(part="id,snippet,replies",
                                             maxResults=max_results, videoId=video_id).execute()
            results = comment_objects.get("items", [])
        except Exception as e:
            print(e)
            return comments

        for item in results: #Iterating over the comments
            # comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            sentence = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            print(sentence)
            filtered_sentence = self.sentence_cleanser.process_sentence(sentence)
            if (len(filtered_sentence.split(' '))>1):
                translated_sentence = self.language_processing_model.convert_language_of_text(filtered_sentence) #Translate the description to english
                cleaned_sentence = self.sentence_cleanser.remove_special_chars(translated_sentence) #Remove any special characters in the comments
                # print(len(cleaned_sentence.split(' ')))
                if (len(cleaned_sentence)>1 and len(cleaned_sentence.split(' '))>2):
                    sentence_type = self.lstm_load.predict_sentence_array([cleaned_sentence]) #Detects if the comment is a question or a general sentence
                    item["snippet"]["topLevelComment"]["snippet"]["processedComments"] = cleaned_sentence
                    item["snippet"]["topLevelComment"]["snippet"]["sentenceType"] = sentence_type[0]['type']
                    if (item["snippet"]["topLevelComment"]["snippet"]["sentenceType"]=='statement'):
                        statements.append(cleaned_sentence)
                    elif (item["snippet"]["topLevelComment"]["snippet"]["sentenceType"]=='question'):
                        questions.append(cleaned_sentence)

                    comments.append(cleaned_sentence)

            reply_count = item['snippet']['totalReplyCount']
            replies = item.get('replies')
            if replies is not None and reply_count != len(replies['comments']):
                #Extract replies of the comment
                replies['comments'] = self.get_comment_replies(self.youtube_object, item['id'], statements, questions, classifier, reply_count)

        print("Comments:\n", "\n".join(comments), "\n")
        print("Questions:\n", "\n".join(questions), "\n")
        return comments

    '''
        Extracts out youtube comments, description and possible question and answer for a given video ID.
        '''

    def youtube_get_comments_enhanced(self, video_id, max_results, statements, questions, classifier, max_results_replies):

        comments = []
        try:
            # Fetchin all the comments
            comment_objects = self.youtube_object.commentThreads().list(part="id,snippet,replies",
                                                                        maxResults=max_results,
                                                                        videoId=video_id).execute()
            results = comment_objects.get("items", [])
        except Exception as e:
            print(e)
            return comments

        for item in results:  # Iterating over the comments
            # comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            sentence = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            print(sentence)
            #Clean Sentence
            filtered_sentence = self.sentence_cleanser.process_sentence(sentence)
            if (len(filtered_sentence.split(' ')) > 1):
                comments.append(filtered_sentence)
                # translated_sentence = self.language_processing_model.convert_language_of_text(
                #     filtered_sentence)  # Translate the description to english
                # cleaned_sentence = self.sentence_cleanser.remove_special_chars(
                #     translated_sentence)  # Remove any special characters in the comments
                # if (len(cleaned_sentence) > 1 and len(cleaned_sentence.split(' ')) > 2):
                #     sentence_type = self.lstm_load.predict_sentence_array(
                #         [cleaned_sentence])  # Detects if the comment is a question or a general sentence
                #     item["snippet"]["topLevelComment"]["snippet"]["processedComments"] = cleaned_sentence
                #     item["snippet"]["topLevelComment"]["snippet"]["sentenceType"] = sentence_type[0]['type']
                #     if (item["snippet"]["topLevelComment"]["snippet"]["sentenceType"] == 'statement'):
                #         statements.append(cleaned_sentence)
                #     elif (item["snippet"]["topLevelComment"]["snippet"]["sentenceType"] == 'question'):
                #         questions.append(cleaned_sentence)
                #
                #     comments.append(cleaned_sentence)


            reply_count = item['snippet']['totalReplyCount']
            replies = item.get('replies')
            # if replies is not None and reply_count != len(replies['comments']):
            if replies is not None:
                # Extract replies of the comment
                replies['comments'] = self.get_comment_replies_enhanced(self.youtube_object, item['id'], statements, questions,
                                                               classifier, reply_count, comments)

        print("Comments:\n", "\n".join(comments), "\n")
        print("Questions:\n", "\n".join(questions), "\n")
        return comments

    '''
    max_results should be multiple of 10
    '''
    def get_comment_replies(self, service, comment_id, statements, questions, classifier, max_results_replies):
        default_size = 10
        request = service.comments().list(
            parentId = comment_id,
            part = 'id,snippet',
            maxResults = min(default_size, max_results_replies)
        )
        replies = []

        while request and max_results_replies>0:
            max_results_replies -= default_size
            response = request.execute()
            reply_list = response['items']
            filtered_texts = filter(lambda filtered_sentence: len(filtered_sentence.split(' ')) > 1, [self.sentence_cleanser.process_sentence(reply["snippet"]["textDisplay"]) for reply in response['items']]) # Get the texts of the replies

            reply_text = list(filter(None, [self.sentence_cleanser.remove_special_chars(self.language_processing_model.convert_language_of_text(reply)) for reply in filtered_texts])) #remove any special characters and remove any empty string
            sentence_type_list = [self.lstm_load.predict_sentence_array([reply_tx]) for reply_tx in reply_text] #Translate the sentences to english
            for reply, each_reply_text, sentence_type in zip(reply_list, reply_text, sentence_type_list):
                reply["snippet"]["sentenceType"] = sentence_type[0]['type']

                if (reply["snippet"]["sentenceType"] == 'statement'):
                    statements.append(each_reply_text)
                elif (reply["snippet"]["sentenceType"] == 'question'):
                    questions.append(each_reply_text)
            replies.extend(reply_list)
            request = service.comments().list_next(request, response) #Fetch the next set of comments

        return replies

    '''
        max_results should be multiple of 10
        '''

    def get_comment_replies_enhanced(self, service, comment_id, statements, questions, classifier, max_results_replies,
                            comments):
        default_size = 10
        request = service.comments().list(
            parentId=comment_id,
            part='id,snippet',
            maxResults=min(default_size, max_results_replies)
        )
        # replies = []

        while request and max_results_replies > 0:
            max_results_replies -= default_size
            response = request.execute()
            reply_list = response['items']
            filtered_texts = filter(lambda filtered_sentence: len(filtered_sentence.split(' ')) > 1,
                                    [self.sentence_cleanser.process_sentence(reply["snippet"]["textDisplay"]) for reply
                                     in response['items']])  # Get the texts of the replies
            comments.extend(filtered_texts)
            # reply_text = list(filter(None, [self.sentence_cleanser.remove_special_chars(self.language_processing_model.convert_language_of_text(reply)) for reply in filtered_texts])) #remove any special characters and remove any empty string
            # sentence_type_list = [self.lstm_load.predict_sentence_array([reply_tx]) for reply_tx in reply_text] #Translate the sentences to english
            # for reply, each_reply_text, sentence_type in zip(reply_list, reply_text, sentence_type_list):
            #     reply["snippet"]["sentenceType"] = sentence_type[0]['type']
            #
            #     if (reply["snippet"]["sentenceType"] == 'statement'):
            #         statements.append(each_reply_text)
            #     elif (reply["snippet"]["sentenceType"] == 'question'):
            #         questions.append(each_reply_text)
            # replies.extend(reply_list)
            # request = service.comments().list_next(request, response) #Fetch the next set of comments

        # return replies

    def fetch_transcript_languages(self, video_id):
        api = YouTubeTranscriptApi()
        list_transcripts = api.list_transcripts(video_id=video_id)
        langs = [transcript.language_code for transcript in list_transcripts]
        return langs

    def fetch_closed_caption(self, video_id, languages):
        api = YouTubeTranscriptApi()
        # list_transcripts = api.list_transcripts(video_id=video_id)
        # api.get_transcripts()
        # transcripts = api.get_transcripts([video_id])
        transcript = api.get_transcript(video_id, languages=languages)
        return transcript

    '''
    Extracts out youtube comments, description and possible question and answers for a give video ID array.
    '''
    def extract_youtube_comments(self, videoIdArray, max_results_comments, max_results_replies):
        classifier = SentenceTypeDetection.getClassifier()
        statements = []
        questions = []
        for videoId in videoIdArray:
            print("Processed video - ", videoId)
            self.youtube_get_comments(videoId, max_results=max_results_comments, statements=statements, questions=questions,
                                           classifier=classifier, max_results_replies=max_results_replies)
        return {"statements": statements, "questions": questions}

    '''
        Extracts out youtube comments, description and possible question and answers for a give video ID array.
        '''

    def extract_youtube_comments_enhanced(self, videoIdArray, max_results_comments, max_results_replies):
        classifier = SentenceTypeDetection.getClassifier()
        statements = []
        questions = []
        texts = []
        for videoId in videoIdArray:
            print("Processed video - ", videoId)
            txt = self.youtube_get_comments_enhanced(videoId, max_results=max_results_comments, statements=statements,
                                               questions=questions,
                                               classifier=classifier, max_results_replies=max_results_replies)

            translated_sentence = [self.language_processing_model.convert_language_of_text(
            phrase) for phrase in txt]
            texts.extend(translated_sentence)

        print("EXTRACTED SENTENCES:")
        print(texts)
        sentence_types = self.lstm_load.predict_sentence_array(texts)
        for text, sentence_type in zip(texts, sentence_types):
            if (len(text) > 1 and len(text.split(' ')) > 2):
                if sentence_type == "question":
                    questions.append(text)
                elif sentence_type == "statement":
                    statements.append(text)
            # except Exception as e:
            #     print("ERROR Getting Sentence Type")
            #     print(e)
            #     print(text)

        return {"statements": statements, "questions": questions}

    def summarize_comments(self, statements, summarizer_model_name):
        if summarizer_model_name not in self.summarizer_model_map:
            raise Exception("Summarizer Model not found!")
        model = self.summarizer_model_map[summarizer_model_name]
        #Clusterize comments, replies and description
        clustered_texts = self.kmeansClusterer.clusterize_texts(statements)
        # print(clustered_texts)
        summary = []
        for each_cluster in clustered_texts:
            joined_texts = '. '.join(each_cluster)
            print(joined_texts)
            summary.append(model.summarizeText(joined_texts))
        return {"summary": summary}

    def answer_questions(self, questions, context, qa_model_name):
        if qa_model_name not in self.ques_ans_model_map:
            raise Exception("Question Answering Model not found!")
        model = self.ques_ans_model_map[qa_model_name]
        context_joined = '. '.join(context)
        answered_questions = [{"question": ques, "answer": model.answer_question(question=ques, context=context_joined)} for ques in filter(lambda ques: len(ques) > 0, questions)]
        return answered_questions


    def wrap_text(self, text):
        return WrapText.wrapText(text)


if __name__ == "__main__":
    # search_text = "kolkata restaurants"
    # videos = youtube_search_keyword('Nature Videos', max_results = 2)
    # videos = youtube_search_keyword(search_text, max_results=2)
    # for video in videos:
    #     print(video["title"])
    #     print("\t", video["videoId"])
    #     print("\t", video["description"])
    ys = YoutubeSearch()
    # videos = ys.youtube_get_videos('Nature Videos', max_results=10)
    # videos = ys.youtube_get_videos('vgfdxfbhlkhvjfcjhjbhjm', max_results=10)
    # videoIdArray = ['viIpUaC6blY']

    query = "vgfdxfbhlkhvjfcjhjbhjm"
    token = "CAoQAA"
    max_results = 10

    # search_keyword = youtube_object.search().list(q=query, part="id, snippet", maxResults=max_results).execute()
    # search_keyword = youtube_object.search().list(pageToken="CAoQAA", part="id, snippet", maxResults=max_results).execute()
    # search_keyword = youtube_object.search().list(pageToken="CAoQAA", part="id, snippet",
    #                                               maxResults=max_results).execute()
    # results = search_keyword.get("items", [])
    #'CAoQAA'


    # videoIdArray = [
    #     # "x0fhGdEc2_Y",
    #     "9HK1ww1HrBU",
    #     "W7laFRcwoBI",
    #     "_jhUvcjElro",
    #     "04UH1iV0CHI",
    #     "xyzGci9Qruc",
    #     "7ZdY_ZbQsh8",
    #     "fjdXERhm6NU",
    #     "7tMEcD0O0lg"
    # ]

    # videoIdArray = [
            # "o0MCsE-04dg",
            #         "CKnGXZxK7zs",
            #         "lr7Jdl2ZexA", "bzLqz6p5HCk", "dQhuZAFoHos"]

    videoIdArray = [
        # "K18nEQf8i9c",
        # "IgAnj6r1O48",
        # "zc_iulZt-us",
        "HWGzQlrJOqM",
        "98kYg52aQeY",
        "wO2SVajcsY4",
        "XSE9dj8t3kE",
        "2vUdkzbXbko",
        "8EeX55DLXVw",
        "ZNEGCogvhwo"
    ]
    videoIdArray = [
        # "Mtyyd9moDeM",
        # "7tMEcD0O0lg",
        # "jojEtcizEMI",
        # "oQJ7cO4t568",
        # "fn4_eItzIT0",
        # "3N4dxGzt6mM",
        # "k6o2Yv0Pi1k",
        # "_BxNQpenzNM",
        # "QnD0Psa8qdE",
        # "YPb3yfR2ssg",
        "WkKsPFBjTkA",
        # "YPb3yfR2ssg"
      ]
    # result = ys.summarize_youtube_comments(videoIdArray, max_results_comments = 2, max_results_replies = 20)
    # result = ys.extract_youtube_comments(videoIdArray, max_results_comments=2, max_results_replies=20)
    # result = {'statements': ['Happy durga puja sir', 'Happy Durga Happy Puja Panchami', 'You probably havent seen Chor Bagan...near. mg metro.. one of finest pandal I bet', 'Dada, wha north Kolkata, another and big Puja visit Will do it, Nav para dada Y Sangha Baranagar. Ehaka thim hei Introduction Look, I guarantee it. ki you like it', 'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day', 'Chorbagan ta top 10 a It would be better to keep it', 'Durga Puja video ', '', 'Kalyani, West Bengal, Nadia district', 'go mom Durga ', 'Jai Maa Durga', 'I say, grandpa, your drone is fine now', 'Patna ka', 'Kharagpur Durga Puja Pandal 2024', 'Hope You Enjoyed The Video Add Me on Social Media Instagram', 'Dhono dhonne puspe vora. It,s poem on rabindra nath thakur.', 'Thanks for watching Add Me on Social Media Instagram', 'Jay maa durga ', 'Dada ami I am saying Shubojit Paul contact a basket ki lures', 'go mom Durga Jai maa Durga ', 'Jay eye di '], 'questions': ['Wishing You Happy Durga Puja', 'Watch my Top 5 Best Durga Puja']}
    # result = {'statements': ['Happy durga puja sir', 'Happy Durga Puja Happy Panchami', 'You may not have seen Chor Bagan...Near Metro...One of the Finest Pandal Bet', 'Dada, you from North Kolkata, will visit another big Puja, now for Dada or Sangha Baranagar. This is the Introduction that I guarantee you will be impressed by the time you watch it.', 'Coming to kolkata on 5th oct, kindly guide us which pandal to visit.. as its last day', 'Those who keep Chorbagans top 10', 'Durga Puja video ', '', 'Kalyani, West Bengal, Nadia district', 'Jai maa Durga ', 'Jai Maa Durga . Har Har Mahadev ', 'Bhai background music download from Katha', 'Hope You Enjoyed The Video Add Me on Social Media Instagram', 'Dhone dhonne puspe vora. Its poem on rabindranath thakur.', 'Thanks for watching Add Me on Social Media Instagram', 'Jay maa durga ', 'Dada I am Shubhjit Paul saying how to contact', 'Jai maa Durga Jai maa Durga ', 'Jay eyes on '], 'questions': ['Wishing You Happy Durga Puja', 'Watch my Top 5 Best Durga Puja', 'helo please make the beautiful procession of maa durga immersion on the streets of kolkata this year on 12 13 and 14 october both north and south kolkata In North kolkata it will mainly take place near hedua park or beadon street 15 20 minutes from hatibagan star theatre but I dont know the way of south kolkata procession please find or search the place where it will take place exactly and do the vlog thank u ...', 'helo please make the beautiful procession of maa durga immersion on the streets of kolkata this year on 12 13 and 14 october both north and south kolkata .. thank u ...']}
    # print(result)
    # print(ys.get_summarizer_model_list())
    # summary = ys.summarize_comments(result["statements"], "Abstractive - T5SmallSummarizer")
    # for smm in summary:
    #     print(smm)

    # qas = ys.answer_questions(result["questions"], '. '.join(result["statements"]), "DistilbertQuestionAnswering")
    # print(qas)
    # x = "a"

    ########################
    # classifier = SentenceTypeDetection.getClassifier()
    # statements = []
    # questions = []
    ys = YoutubeSearch()
    # videos = ys.youtube_get_videos_by_token(token, max_results)
    extracted_texts = ys.extract_youtube_comments(videoIdArray, max_results_comments = 10, max_results_replies = 20)
    # extracted_texts = ys.extract_youtube_comments_enhanced(videoIdArray, max_results_comments=10, max_results_replies=20)
    print("EXTRCATED - ")
    print(extracted_texts)

    extracted_texts = [
        'Thriller Ba Horror, Audio Story Gula Sunte Onk Valo Lage Amar.Ovinoy r thriller station er audio er moto eto crystal clear sound er sathe background music er combination, and jara voice den tader moto sundor vabe upostapon kono chano chano chano radio shadio shady.kothaw pai nai.Na .. Ekta kmn niramis niramish typer lage amr kache personally .. Stoty jotoi romamcokor hok na kno sunte gele r Valo Lge na.Sunechi.bepar .. tar poreo request yhakbe apnara jodi r ekyu taratari story dite paren ... thanks thriller station ..',
        'Excellent story &amp; presentation',
        'Ashonkhyo dhonyobaad o bhalobasha apnake! Credit goes to the sound designer babu sarkar for the zd audio effCTs & amp;Quality!amader channel ti ke egiye niye jte sahajyo korben! 🙂🙂🤟🤟',
        'Thanx a lot!Plz do like,share,subscribe n help our channel to grow!🙂🙂🤟🤟', 'Think this is nice, think awesome',
        'Radio Marchi for Moto Valona', 'Excellent story &amp; presentation',
        'Ashonkhyo dhonyobaad o bhalobasha apnake! Credit goes to the sound designer babu sarkar for the zd audio effCTs & amp;Quality!amader channel ti ke egiye niye jte sahajyo korben! 🙂🙂🤟🤟',
        'Thanx a lot!Plz do like,share,subscribe n help our channel to grow!🙂🙂🤟🤟', 'Think this is nice, think awesome',
        'Radio Marchi for Moto Valona',
        'Each one of you performs beautiful and thank everyone.Especially the voice and acting of Raja Bhattacharya is imperative.Want to hear more stories.',
        'Malda 🙂', 'Darun', 'Mr. Haldar moshai is so generous and have big brave heart...salute him',
        'Sound ears in the sound, oh boring', 'Colonial voice very good']

    # sentence_types = ys.lstm_load.predict_sentence_array(extracted_texts)
    # x = "a"
    # comments = ys.youtube_get_comments('viIpUaC6blY', max_results = 2, statements = statements, questions = questions, classifier = classifier, max_results_replies = 20)
    # print(comments)
    # wrapped_text = WrapText.wrapText(statements)
    # ts = TextSummarization.TextSummarization()
    # summary = ts.summarizeText(wrapped_text)
    # print(summary)
    # print(statements)
    # print(questions)

    ##########################
    # languages = fetch_transcript_languages('_QRjakwgguI')
    # print(languages)
    # transcript = fetch_closed_caption('_QRjakwgguI', languages)
    # print(list_transcripts)

    # for line in transcript:
    #     print(line)
    # cleaned_sentence = [
    #     "Original Voice ",
    #     "What year did Dell announce its plans to buy its building?",
    #     "Original Voice"
    # ]
    # asasa = ys.remove_spaces("Original Voice ")
    # print(asasa)
    # print(len(asasa.split(' ')))
    # sentence_type = ys.lstm_load.predict_sentence_array(cleaned_sentence)
    # print(sentence_type)
#Summarize text
# Extract answers for the questions.