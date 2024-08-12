from apiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

from DataProcessing.SentenceCleanser import SentenceCleanser
from LanguageDetectorMain.LanguageDetectorMain import LanguageDetectorMain
from SentenceDetectionGeneratorDetector import SentenceTypeDetection
from DataProcessing import TextSummarizer, WrapText

DEVELOPER_KEY = "AIzaSyDIyibF6V6UU4ctTjlojI9sI113AJ01y20"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube_object = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                                        developerKey = DEVELOPER_KEY)

class YoutubeSearch:

    def __init__(self):
        self.language_processing_model = LanguageDetectorMain()
        self.sentence_cleanser = SentenceCleanser()
        self.text_summarizer = TextSummarizer.TextSummarizer()

    def youtube_get_videos(self, query, max_results):
        search_keyword = youtube_object.search().list(q = query, part = "id, snippet",
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

        return videos

    def youtube_get_comments(self, video_id, max_results, statements, questions, classifier, max_results_replies):

        comments = []
        try:
            comment_objects = youtube_object.commentThreads().list(part="id,snippet,replies",
                                             maxResults=max_results, videoId=video_id).execute()
            results = comment_objects.get("items", [])
        except Exception as e:
            print(e)
            return comments

        for item in results:
            # comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            sentence = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            print(sentence)
            filtered_sentence = self.sentence_cleanser.process_sentence(sentence)
            if (len(filtered_sentence.split(' '))>1):
                translated_sentence = self.language_processing_model.detect_language_of_text(filtered_sentence)
                cleaned_sentence = self.sentence_cleanser.remove_special_chars(translated_sentence)
                sentence_type = SentenceTypeDetection.sentenceDetectionModel([cleaned_sentence], classifier)
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
                replies['comments'] = self.get_comment_replies(youtube_object, item['id'], statements, questions, classifier, max_results_replies)

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

            # filtered_sentence = self.sentence_cleanser.process_sentence(sentence)
            # translated_sentence = self.language_processing_model.detect_language_of_text(filtered_sentence)
            # cleaned_sentence = self.sentence_cleanser.remove_special_chars(translated_sentence)
            # sentence_type = SentenceTypeDetection.sentenceDetectionModel([cleaned_sentence], classifier)
            # len(filtered_sentence.split(' ')) > 1
            filtered_texts = filter(lambda filtered_sentence: len(filtered_sentence.split(' ')) > 1, [self.sentence_cleanser.process_sentence(reply["snippet"]["textDisplay"]) for reply in response['items']])
            # reply_text = [self.sentence_cleanser.remove_special_chars(self.language_processing_model.detect_language_of_text((self.sentence_cleanser.process_sentence(reply["snippet"]["textDisplay"])))) for reply in response['items']]
            reply_text = [self.sentence_cleanser.remove_special_chars(self.language_processing_model.detect_language_of_text(reply)) for reply in filtered_texts]
            sentence_type_list = SentenceTypeDetection.sentenceDetectionModel(reply_text, classifier)
            for reply, each_reply_text, sentence_type in zip(reply_list, reply_text, sentence_type_list):
                reply["snippet"]["sentenceType"] = sentence_type['type']
                # sentence = reply["snippet"]["textDisplay"]
                # sentence_type = SentenceTypeDetection.TestSentenceDetectionModel(reply["snippet"]["textDisplay"])
                # reply["snippet"]["sentenceType"] = sentence_type
                if (sentence_type['type'] == 'statement'):
                    statements.append(each_reply_text)
                elif (sentence_type['type'] == 'question'):
                    questions.append(each_reply_text)
            replies.extend(reply_list)
            request = service.comments().list_next(request, response)

        return replies

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

    def summarize_youtube_comments(self, videoIdArray, max_results_comments, max_results_replies):
        classifier = SentenceTypeDetection.getClassifier()
        statements = []
        questions = []
        for videoId in videoIdArray:
            comments = self.youtube_get_comments(videoId, max_results=max_results_comments, statements=statements, questions=questions,
                                           classifier=classifier, max_results_replies=max_results_replies)

        wrapped_text = WrapText.wrapText(statements)
        # ts = TextSummarizer.TextSummarizer()
        summary = self.text_summarizer.summarizeText(wrapped_text)
        result = {}
        result["summary"] = summary[0]["summary_text"]
        answered_questions = [{"question": ques, "answer": self.text_summarizer.answer_question(question=ques, context=wrapped_text)} for ques in questions]
        result["questions"] = answered_questions
        return result


if __name__ == "__main__":
    search_text = "kolkata restaurants"
    # videos = youtube_search_keyword('Nature Videos', max_results = 2)
    # videos = youtube_search_keyword(search_text, max_results=2)
    # for video in videos:
    #     print(video["title"])
    #     print("\t", video["videoId"])
    #     print("\t", video["description"])
    ys = YoutubeSearch()
    videoIdArray = ['viIpUaC6blY']
    result = ys.summarize_youtube_comments(videoIdArray, max_results_comments = 2, max_results_replies = 20)
    print(result)
    ########################
    # classifier = SentenceTypeDetection.getClassifier()
    # statements = []
    # questions = []
    # ys = YoutubeSearch()
    # comments = ys.youtube_get_comments('viIpUaC6blY', max_results = 2, statements = statements, questions = questions, classifier = classifier, max_results_replies = 20)
    # print(comments)
    # wrapped_text = WrapText.wrapText(statements)
    # ts = TextSummarizer.TextSummarizer()
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

#Summarize text
# Extract answers for the questions.