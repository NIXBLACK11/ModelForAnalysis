import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from analyse import analyseVideo

def get_additional_details(genre):
    recommendations = {
        'MrBeastType': {
            'Visuals': "Your video should have dynamic and energetic editing. Include on-screen graphics, counters, and engaging visual effects. Incorporate a mix of close-ups, wide shots, and creative camera angles. Aim for high-quality production with attention to detail.",
            'Audio': "For audio, use energetic background music that complements the fast-paced visuals. Ensure clear and expressive commentary or dialogue. Strategically use sound effects to enhance the overall impact."
        },
        'VlogType': {
            'Visuals': "Create a video that follows the daily life or experiences of the creator. Include a mix of handheld shots, personal moments, and travel footage. Focus on authentic and candid visual storytelling.",
            'Audio': "For audio, add background music that complements the mood and atmosphere. Aim for conversational and natural dialogue. Include ambient sounds and noises from the surroundings for an immersive experience."
        },
        'TechReviewType': {
            'Visuals': "Your video should focus on showcasing and reviewing technology products. Include detailed shots of gadgets, interfaces, and demonstrations. Maintain clean and professional visual aesthetics with product close-ups and technical visual elements.",
            'Audio': "For audio, provide clear and informative commentary on the technology being reviewed. Use minimal background music to maintain focus on information. Incorporate sound effects relevant to the tech products being showcased."
        },
        'GamingType': {
            'Visuals': "Feature gameplay footage, often with facecam or reactions. Use in-game graphics, HUD elements, and dynamic camera angles. Consider adding overlays with live chat, alerts, or game-related information.",
            'Audio': "For audio, include commentary or live reactions to the gameplay. Enhance the gaming experience with in-game sound effects and music. Add background music or soundtracks specific to the game for atmosphere."
        },
        'MinimalistType': {
            'Visuals': "Emphasize simplicity and clean aesthetics. Limit the use of on-screen elements and effects. Feature uncluttered backgrounds and minimalistic design.",
            'Audio': "For audio, use subtle background music or ambient sounds. Keep commentary or narration clear and concise. Minimize the use of sound effects unless essential for the content."
        }
    }

    return recommendations.get(genre, {'Visuals': '', 'Audio': ''})


def get_genre_details(genre):
    genre_details = {
        'MrBeastType': {
            'Visuals': "Your video has dynamic and energetic editing. Features on-screen graphics, counters, and engaging visual effects. Incorporates a mix of close-ups, wide shots, and creative camera angles. High-quality production with attention to detail.",
            'Audio': "The audio in your video is accompanied by energetic background music that complements the fast-paced visuals. Clear and expressive commentary or dialogue adds to the overall impact. Sound effects are strategically placed to enhance the excitement of challenges or stunts."
        },
        'VlogType': {
            'Visuals': "Your video follows the daily life or experiences of the creator. May include a mix of handheld shots, personal moments, and travel footage. Relies on authentic and candid visual storytelling.",
            'Audio': "The audio in your video is accompanied by background music that sets the mood and atmosphere. Conversational and natural dialogue creates an engaging narrative. Ambient sounds and noises from the surroundings contribute to the overall immersive experience."
        },
        'TechReviewType': {
            'Visuals': "Your video focuses on showcasing and reviewing technology products. Includes detailed shots of gadgets, interfaces, and demonstrations. Clean and professional visual aesthetics. Product close-ups and technical visual elements.",
            'Audio': "The audio in your video features clear and informative commentary on the technology being reviewed. Minimal background music maintains focus on information. Sound effects are used to highlight key aspects of the tech products being showcased."
        },
        'GamingType': {
            'Visuals': "Your video features gameplay footage, often with facecam or reactions. In-game graphics, HUD elements, and dynamic camera angles. May include overlays with live chat, alerts, or game-related information.",
            'Audio': "The audio in your video includes commentary or live reactions to the gameplay. In-game sound effects and music enhance the gaming experience. Background music or soundtracks specific to the game add to the overall atmosphere."
        },
        'MinimalistType': {
            'Visuals': "Your video emphasizes simplicity and clean aesthetics. Limited use of on-screen elements and effects. Often features uncluttered backgrounds and minimalistic design.",
            'Audio': "The audio in your video is characterized by subtle background music or ambient sounds. Clear and concise commentary or narration complements the minimalist visuals. Minimal use of sound effects adds to the content's simplicity."
        }
    }

    return genre_details.get(genre, {'Visuals': '', 'Audio': ''})

def callAnalyse(video_path, videoGenre):
    
    # res = analyseVideo(video_path)
    res = {
        'genre_counts': {'MrBeastType': 4, 'VlogType': 3, 'TechReviewType': 94, 'GamingType': 2, 'MinimalistType': 7},
        'genre_counts_audio': {'MrBeastType': 5, 'VlogType': 0, 'TechReviewType': 0, 'GamingType': 0, 'MinimalistType': 0}
    }
    total_video = sum(res['genre_counts'].values())
    total_audio = sum(res['genre_counts_audio'].values())

    percentage_video = {genre: count / total_video * 100 for genre, count in res['genre_counts'].items()}
    percentage_audio = {genre: count / total_audio * 100 for genre, count in res['genre_counts_audio'].items()}

    ans = {
        'visual': {},
        'audio': {},
        'details': {}
    }

    # Check and add details for video analysis
    for genre, percentage in percentage_video.items():
        if percentage != 0:
            ans['visual'][genre.lower()] = {
                'percentage': percentage,
                'details': get_genre_details(genre)['Visuals']
            }

    # Check and add details for audio analysis
    for genre, percentage in percentage_audio.items():
        if percentage != 0:
            ans['audio'][genre.lower()] = {
                'percentage': percentage,
                'details': get_genre_details(genre)['Audio']
            }

    detected_genre_details = get_additional_details(videoGenre)
    ans['details'][videoGenre.lower()] = {
        'whatShouldBe': f"The data was trained on trending videos in this genre so based on that your video should have:{detected_genre_details}"
    }

    return ans
