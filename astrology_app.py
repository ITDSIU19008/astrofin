import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import swisseph as swe
from datetime import datetime
from geopy.geocoders import Nominatim
import pytz
import streamlit as st
import re
from timezonefinder import TimezoneFinder
import geopy.geocoders
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

# Đường dẫn tương đối tới thư mục ephemeris (trong cùng thư mục với file Python chính)
relative_ephe_path = os.path.join(os.path.dirname(__file__), 'sweph')
# Đặt đường dẫn tới thư viện Swiss Ephemeris
swe.set_ephe_path(relative_ephe_path)

# Đọc dữ liệu từ các file CSV
financial_traits_path = 'combined_financial_traits.csv'
keyword_to_trait_mapping_df = pd.read_csv('keyword_to_trait_mapping.csv')
product_keywords_path = 'product_keywords.csv'
aspect_path = 'aspect_sc.csv'

# Read data
financial_traits_df = pd.read_csv(financial_traits_path)
keyword_to_trait_mapping = keyword_to_trait_mapping_df.set_index('Keyword').T.to_dict()
product_keywords_df = pd.read_csv(product_keywords_path)
product_keywords = product_keywords_df.set_index('Product').T.to_dict('list')
aspect_influence_factors_df = pd.read_csv(aspect_path)

# Rulers for each zodiac sign
rulers = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": ["Mars", "Pluto"],
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": ["Saturn", "Uranus"],
    "Pisces": ["Jupiter", "Neptune"]
}
# Ảnh hưởng của hành tinh đối với hành vi tài chính dựa trên khoảng cách từ các hành tinh
planet_impacts = {
    'Venus': 0.20,
    'Jupiter': 0.15,
    'Saturn': 0.15,
    'Mars': 0.10,
    'Pluto': 0.10,
    'Mercury': 0.05
}

# Vị trí của các hành tinh trong cung Hoàng Đạo
zodiac_positions = {
    'Sun': 10,  
    'Moon': 7,
    'Mercury': 5,
    'Mars': 8,
    'Venus': 6,
    'Jupiter': 9,
    'Saturn': 7,
    'Uranus': 4,
    'Neptune': 3,
    'Pluto': 2
}

# Function to calculate zodiac sign and degree
def get_zodiac_sign_and_degree(degree):
    zodiacs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    sign_index = int(degree / 30)
    sign = zodiacs[sign_index]
    degree_in_sign = degree % 30
    return sign, degree_in_sign

def get_city_suggestions(query):
    geolocator = Nominatim(user_agent="astrology_app")
    location = geolocator.geocode(query, exactly_one=False, limit=5, language='en')  # Thêm tham số 'language'
    if location:
        return [f"{loc.address} ({loc.latitude}, {loc.longitude})" for loc in location]
    return []

# Function to convert decimal degrees to DMS (degrees, minutes, seconds)
def decimal_to_dms(degree):
    d = int(degree)
    m = int((degree - d) * 60)
    s = (degree - d - m / 60) * 3600
    return f"{d}° {m}' {s:.2f}\""

# Function to get latitude, longitude, and timezone from place name
def get_location_and_timezone(place):
    geolocator = geopy.geocoders.Nominatim(user_agent="astrology_app")
    location = geolocator.geocode(place)
    if location:
        lat, lon = location.latitude, location.longitude
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lat=lat, lng=lon)
        return lat, lon, timezone
    else:
        st.error(f"Cannot find location for place: {place}")
        return None #10.8231, 106.6297, 'Asia/Ho_Chi_Minh'  # Default to Ho Chi Minh

# Function to calculate planetary positions
def get_planet_positions(year, month, day, hour, minute, lat, lon, timezone):
    tz = pytz.timezone(timezone)
    local_datetime = datetime(year, month, day, hour, minute)
    local_datetime = tz.localize(local_datetime)
    utc_datetime = local_datetime.astimezone(pytz.utc)
    jd = swe.julday(utc_datetime.year, utc_datetime.month, utc_datetime.day, utc_datetime.hour + utc_datetime.minute / 60.0)
    
    swe.set_topo(lon, lat, 0)
    planets = {
        'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY, 'Venus': swe.VENUS,
        'Mars': swe.MARS, 'Jupiter': swe.JUPITER, 'Saturn': swe.SATURN,
        'Uranus': swe.URANUS, 'Neptune': swe.NEPTUNE, 'Pluto': swe.PLUTO,
        'North Node': swe.TRUE_NODE, 'Chiron': swe.CHIRON, 'Lilith': swe.MEAN_APOG,
        'True Lilith': swe.OSCU_APOG, 'Vertex': swe.VERTEX, 'Ceres': swe.CERES,
        'Pallas': swe.PALLAS, 'Juno': swe.JUNO, 'Vesta': swe.VESTA
    }
    
    positions = {}
    for planet, id in planets.items():
        position, _ = swe.calc_ut(jd, id)
        positions[planet] = position[0]
    
    positions['South Node'] = (positions['North Node'] + 180.0) % 360.0
    return positions

# Function to calculate houses and ascendant
def calculate_houses_and_ascendant(year, month, day, hour, minute, lat, lon, timezone):
    tz = pytz.timezone(timezone)
    local_datetime = datetime(year, month, day, hour, minute)
    local_datetime = tz.localize(local_datetime)
    utc_datetime = local_datetime.astimezone(pytz.utc)
    jd = swe.julday(utc_datetime.year, utc_datetime.month, utc_datetime.day,
                    utc_datetime.hour + utc_datetime.minute / 60.0 + utc_datetime.second / 3600.0)
    
    # House calculation using Placidus system
    houses, ascmc = swe.houses(jd, lat, lon, b'P')
    ascendant = ascmc[0]
    house_cusps = {f'House {i+1}': houses[i] for i in range(12)}
    return ascendant, house_cusps

# Function to determine which house a planet is in
def get_house_for_planet(planet_degree, houses):
    house_degrees = [houses[f'House {i+1}'] for i in range(12)]
    house_degrees.append(house_degrees[0] + 360)  # wrap-around for the 12th house
    
    for i in range(12):
        start = house_degrees[i]
        end = house_degrees[(i+1) % 12]
        if start < end:
            if start <= planet_degree < end:
                return i + 1  # Trả về số nhà thay vì chuỗi "House X"
        else:  # handle the wrap-around
            if start <= planet_degree or planet_degree < end:
                return i + 1  # Trả về số nhà thay vì chuỗi "House X"
    return 'Unknown'  # Default case if no house matches


# Function to display planetary data with house placement
def create_astrology_dataframe(positions, houses, ascendant):
    planets_info = []

    # Add Ascendant to the list
    asc_sign, asc_degree_in_sign = get_zodiac_sign_and_degree(ascendant)
    asc_dms = decimal_to_dms(asc_degree_in_sign)
    
    # Append Ascendant first (no need for house ruler)
    planets_info.append(["Ascendant", asc_sign, "1", asc_dms])  # Chỉ 4 giá trị

    # Append other planets
    for planet, degree in positions.items():
        sign, degree_in_sign = get_zodiac_sign_and_degree(degree)
        dms = decimal_to_dms(degree_in_sign)
        house_cup = get_house_for_planet(degree, houses)  # Chỉ lấy house cup thay vì house ruler
        planets_info.append([planet, sign, house_cup, dms])  # Chỉ 4 giá trị

    # Đảm bảo là có đúng 4 cột
    df = pd.DataFrame(planets_info, columns=['Planet', 'Zodiac Sign', 'House', 'Degree'])
    return df

def create_house_dataframe(houses, ascendant):
    house_info = []

    # Xác định cung hoàng đạo và độ của Ascendant
    asc_sign, asc_degree_in_sign = get_zodiac_sign_and_degree(ascendant)
    asc_dms = decimal_to_dms(asc_degree_in_sign)
    asc_house_ruler = rulers.get(asc_sign, "Unknown")
    if isinstance(asc_house_ruler, list):
        asc_house_ruler = ', '.join(asc_house_ruler)
    
    # Append Ascendant with Zodiac Sign, Ruler, and Degree
    house_info.append(['Ascendant', asc_sign, asc_house_ruler, asc_dms])  # Sử dụng Ascendant cho nhà 1

    # Append other houses with Zodiac Sign and Ruler
    for i in range(1, 12):
        house_degree = houses[f'House {i+1}']
        house_sign, degree_in_sign = get_zodiac_sign_and_degree(house_degree)
        dms = decimal_to_dms(degree_in_sign)
        house_ruler = rulers.get(house_sign, "Unknown")
        if isinstance(house_ruler, list):
            house_ruler = ', '.join(house_ruler)
        house_info.append([f'House {i+1}', house_sign, house_ruler, dms])

    # Tạo DataFrame từ house_info, bao gồm cột House, House Cup, Ruler và Degree
    df = pd.DataFrame(house_info, columns=['House', 'House Cup', 'Ruler', 'House Cup Degree'])
    return df

# Hàm để tính các góc hợp giữa các hành tinh và góc độ
def calculate_aspects(positions):
    aspects = []
    planets = list(positions.keys())
    aspect_angles = [0, 60, 90, 120, 150, 180]
    aspect_names = ['Conjunction', 'Sextile', 'Square', 'Trine', 'Quincunx', 'Opposition']
    aspect_orbs = [10.5, 6.1, 7.8, 8.3, 2.7, 10]  # Orb for each aspect

    for i, planet1 in enumerate(planets):
        for planet2 in planets[i + 1:]:
            pos1 = positions[planet1]
            pos2 = positions[planet2]
            angle = abs(pos1 - pos2)
            if angle > 180:
                angle = 360 - angle
            for aspect_angle, orb, name in zip(aspect_angles, aspect_orbs, aspect_names):
                if abs(angle - aspect_angle) <= orb:
                    aspects.append((planet1, planet2, name, round(angle, 2)))  # Lưu cả góc độ (angle) đã được làm tròn
                    break
    return aspects



# Hàm để tạo DataFrame cho các góc hợp của người dùng, bổ sung góc độ
def create_aspects_dataframe(aspects, positions):
    aspect_data = []
    for planet1, planet2, aspect_type, degree in aspects:
        # Lấy cung hoàng đạo và độ của hành tinh 1
        planet1_sign, planet1_degree_in_sign = get_zodiac_sign_and_degree(positions[planet1])
        # Lấy cung hoàng đạo và độ của hành tinh 2
        planet2_sign, planet2_degree_in_sign = get_zodiac_sign_and_degree(positions[planet2])
        
        # Thêm dữ liệu vào bảng
        aspect_data.append([planet1, planet1_sign, planet1_degree_in_sign, planet2, planet2_sign, planet2_degree_in_sign, aspect_type, degree])
    
    # Tạo DataFrame với các cột cần thiết
    df = pd.DataFrame(aspect_data, columns=['Planet 1', 'Zodiac Sign 1', 'Degree in Sign 1', 'Planet 2', 'Zodiac Sign 2', 'Degree in Sign 2', 'Aspect', 'Degree'])
    
    return df


# --------------------------------TRAITs-----------------------------------------------
# Hàm xác định hành tinh dẫn dắt
def get_dominant_planet(planet1, planet2, individual_planets):
    if planet1 not in zodiac_positions or planet2 not in zodiac_positions:
        return None  # Bỏ qua nếu hành tinh không trong danh sách

    planet1_power = zodiac_positions.get(planet1, 0)
    planet2_power = zodiac_positions.get(planet2, 0)

    try:
        planet1_sign = [sign for planet, sign in individual_planets if planet == planet1][0]
    except IndexError:
        planet1_sign = None
    
    try:
        planet2_sign = [sign for planet, sign in individual_planets if planet == planet2][0]
    except IndexError:
        planet2_sign = None

    if planet1_sign and (planet1 in rulers.get(planet1_sign, []) if isinstance(rulers.get(planet1_sign), list) else planet1 == rulers.get(planet1_sign)):
        return planet1

    if planet2_sign and (planet2 in rulers.get(planet2_sign, []) if isinstance(rulers.get(planet2_sign), list) else planet2 == rulers.get(planet2_sign)):
        return planet2

    if planet1_power > planet2_power:
        dominant_planet = planet1
    elif planet2_power > planet1_power:
        dominant_planet = planet2
    else:
        dominant_planet = planet1
    
    return dominant_planet
# Định dạng các góc hợp từ bảng aspects
def format_aspects(row, individual_planets):
    # Lấy nội dung từ cột 'Aspects'
    aspects = row['Aspects']
    
    # Sử dụng regular expression để trích xuất planet và aspect type
    aspects_list = df_aspects[['Planet 1', 'Planet 2', 'Aspect']].values.tolist()
    formatted_aspects = []
    for planet1, planet2, aspect_type in aspects_list:
        if planet1 not in zodiac_positions or planet2 not in zodiac_positions:
            continue
        dominant_planet = get_dominant_planet(planet1, planet2, individual_planets)
        if dominant_planet == planet2:
            planet1, planet2 = planet2, planet1
        formatted_aspects.append(f"{planet1} {aspect_type} {planet2}")
    return "\n".join(formatted_aspects)

# Trích xuất các góc hợp liên quan đến các hành tinh đã chọn
def extract_relevant_aspects(formatted_aspects, relevant_planets):
    aspects = re.findall(r"(\w+)\s+(Conjunction|Sextile|Square|Trine|Opposition|Quincunx)\s+(\w+)", formatted_aspects)
    
    # Chỉ giữ lại các góc hợp liên quan đến các hành tinh trong relevant_planets
    filtered_aspects = []
    for aspect in aspects:
        planet1, aspect_type, planet2 = aspect
        if planet1 in relevant_planets and planet2 in relevant_planets:
            filtered_aspects.append((planet1, aspect_type, planet2))
        else:
            print(f"Aspect không hợp lệ: {planet1} hoặc {planet2}")
    
    return filtered_aspects

# Tính toán các đặc điểm tài chính dựa trên vị trí hành tinh và góc hợp
def calculate_financial_traits(individual_planets, formatted_aspects):
    # Danh sách các hành tinh mà bạn muốn lấy
    selected_planets = ['Sun', 'Moon', 'Mercury', 'Mars', 'Venus', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    individual_planets = [(row['Planet'], row['Zodiac Sign']) for _, row in df_positions.iterrows() if row['Planet'] in selected_planets]

    final_scores = {trait: 0 for trait in ['Adventurous', 'Convenience', 'Impulsive', 'Conservative', 'Caution', 'Analytical']}

    # Tính điểm dựa trên vị trí của các hành tinh và cung hoàng đạo
    for planet, sign in individual_planets:
        planet_scores_df = financial_traits_df[financial_traits_df['Planet'] == f"{planet} {sign}"]
        if not planet_scores_df.empty:
            planet_scores = planet_scores_df.iloc[0]
            # print(planet_scores)

            for trait in final_scores.keys():
                base_score = planet_scores[trait]
                
                if planet in planet_impacts:
                    base_score += planet_impacts[planet]
                    # print(base_score)

                final_scores[trait] += base_score
    #             print(final_scores)
    # print(final_scores)

    # Khởi tạo tổng ảnh hưởng của các góc hợp cho từng đặc điểm
    aspects = extract_relevant_aspects(formatted_aspects, [planet[0] for planet in individual_planets])
    # print(f"Extracted Aspects: {aspects}")
    
     # Initialize scores
    final_scores = {trait: 0 for trait in ['Adventurous', 'Convenience', 'Impulsive', 'Conservative', 'Caution', 'Analytical']}

    # Compute base scores from planets and zodiac signs
    for planet, sign in individual_planets:
        planet_scores_df = financial_traits_df[financial_traits_df['Planet'] == f"{planet} {sign}"]
        if not planet_scores_df.empty:
            planet_scores = planet_scores_df.iloc[0]
            for trait in final_scores.keys():
                base_score = planet_scores[trait]
                if planet in planet_impacts:
                    base_score += planet_impacts[planet]
                    # print(base_score)
                final_scores[trait] += base_score
    #             print(final_scores)
    # print(final_scores)

    # Initialize total aspect influence
    aspects = extract_relevant_aspects(formatted_aspects, [planet[0] for planet in individual_planets])
    total_aspect_influence = {trait: 0 for trait in final_scores.keys()}

    # Processing aspects from the CSV and checking if the data is being matched
    for _, row in df_aspects.iterrows():
        planet1, planet2, aspect_type = row['Planet 1'], row['Planet 2'], row['Aspect']
        dominant_planet = get_dominant_planet(planet1, planet2, individual_planets)
        
        if dominant_planet == planet2:
            planet1, planet2 = planet2, planet1

        # Check for matching aspect
        matching_aspect_df = aspect_influence_factors_df[
            (aspect_influence_factors_df['Planet1'] == planet1) &
            (aspect_influence_factors_df['Planet2'] == planet2) &
            (aspect_influence_factors_df['Aspect'] == aspect_type)
        ]
        
        if not matching_aspect_df.empty:
            for trait in final_scores.keys():
                aspect_influence_value = matching_aspect_df.iloc[0].get(trait)
                if aspect_influence_value:
                    aspect_influence = float(aspect_influence_value.strip('%')) / 100
                    total_aspect_influence[trait] += aspect_influence
                    # print(total_aspect_influence)


    # Apply aspect influence to the final scores
    for trait in final_scores.keys():
        adjusted_score = final_scores[trait] + (total_aspect_influence[trait] * 10)
        final_scores[trait] = adjusted_score

    # Normalize final scores to be between 0 and 5
    for trait in final_scores.keys():
        final_scores[trait] /= len(individual_planets)
        final_scores[trait] = min(max(final_scores[trait], 0), 5)
    print(final_scores)
    return final_scores


# -------------------------------------DRAW RADAR CHART----------------------------------

# Định nghĩa hàm để xác định mức độ dựa trên điểm
def get_score_level(score, language="Tiếng Việt"):
    if language == "Tiếng Việt":
        if 1.00 <= score <= 1.80:
            return "Rất thấp"
        elif 1.81 <= score <= 2.60:
            return "Thấp"
        elif 2.61 <= score <= 3.40:
            return "Trung bình"
        elif 3.41 <= score <= 4.20:
            return "Cao"
        elif 4.21 <= score <= 5.00:
            return "Rất cao"
    elif language == "English":
        if 1.00 <= score <= 1.80:
            return "Incredibly low"
        elif 1.81 <= score <= 2.60:
            return "Low"
        elif 2.61 <= score <= 3.40:
            return "Average"
        elif 3.41 <= score <= 4.20:
            return "High"
        elif 4.21 <= score <= 5.00:
            return "Incredibly high"


# Function to get hover text color based on score level
def get_score_color(score, language="Tiếng Việt"):
    level = get_score_level(score, language)
    if level in ["Rất thấp", "Incredibly low"]:
        return "#ff0000"  # Pastel Red
    elif level in ["Thấp", "Low"]:
        return "#ff5f00"  # Pastel Orange
    elif level in ["Trung bình", "Average"]:
        return "#ffaf00"  # Pastel Yellow
    elif level in ["Cao", "High"]:
        return "#008700"  # Pastel Green
    else:  # "Rất cao" or "Incredibly high"
        return "#005fff"  # Pastel Blue
        
# Hàm vẽ radar chart tương tác với plotly
def plot_radar_chart(final_scores, average_scores):
    traits = list(final_scores.keys())
    scores = [final_scores[trait] for trait in traits]
    avg_scores = [average_scores[trait] for trait in traits]
    
    # Bổ sung giá trị đầu tiên vào cuối để tạo vòng tròn khép kín
    traits += [traits[0]]
    scores += [scores[0]]
    avg_scores += [avg_scores[0]]

    # Tạo radar chart với plotly
    fig = go.Figure()

    # Tạo dữ liệu hover với cả điểm và mức độ của từng trait
    hover_texts_avg = [f"Score: {score:.2f}<br>Level: {get_score_level(score)}" for score in avg_scores]
    hover_texts_user = [f"Score: <b>{score:.2f}</b><br>Level: <b>{get_score_level(score, language)}</b>" for score in scores]

    # Thêm đường của Average Scores với thông tin hover
    fig.add_trace(go.Scatterpolar(
        r=avg_scores,
        theta=traits,
        fill='toself',
        name='Average Scores',
        line=dict(color='rgba(0, 0, 255, 0.35)', dash='dashdot'),  # A lighter blue to simulate transparency
        fillcolor='rgba(204, 204, 255, 0.35)',
        hoverinfo='text',
        # hovertext=hover_texts_avg  # Hiển thị hover text
    ))


    # Thêm đường của Your Scores với thông tin hover
    fig.add_trace(go.Scatterpolar(
    r=scores,
    theta=traits,
    fill='toself',
    name='<span style="color:black;">Your Trait</span>',
    line=dict(color='orange'),
    fillcolor='rgba(255, 165, 0, 0.25)',
    hoverinfo='text',
    hovertext=hover_texts_user,  # Display hover text
    marker=dict(
        size=9  # Increase the size of the dots
        # color='orange',  # Color of the dots
    )
))


    # Cài đặt layout được tùy chỉnh để cải thiện bố cục
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Nền của chính radar chart
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Nền của toàn biểu đồ   
        polar=dict(
            bgcolor="rgba(0, 0, 0, 0)",  # Nền trong suốt hoặc màu sắc nhẹ
            radialaxis=dict(
                visible=True,
                range=[0, 5],  # Giữ lại range từ 0-5
                showticklabels=False,  # Ẩn nhãn tick
                ticks='',
                gridcolor="#756075",  # Màu lưới nhẹ
                gridwidth=1.5,  # Độ dày lưới vừa phải
                griddash='dashdot',
                linecolor="rgba(0, 0, 0, 0)",
            ),
            angularaxis=dict(
                visible=True,
                tickfont=dict(size=12, color="rgba(150, 150, 255, 1)", family="Times"),  # Kích thước font cho nhãn
                rotation=45,  # Xoay nhãn để trông đỡ bị chồng chéo
                direction="clockwise",  # Điều chỉnh hướng nhãn
                linewidth=1.5,  # Độ dày của trục góc
                gridcolor="#756075",
                linecolor="#756075" , # Đặt màu trắng để lưới góc không quá rõ
            ),
        ),
        showlegend=True,  # Hiển thị chú thích
        hoverlabel=dict(
            bgcolor="white",  # Nền trắng
            font_size=16,  # Kích thước chữ to hơn
            font_color="black"  # Màu chữ đen
        ),
        font=dict(size=12),  # Kích thước font tổng thể
        margin=dict(l=40, r=40, b=40, t=40),  # Điều chỉnh lề cho cân đối
        # paper_bgcolor="rgba(240, 240, 240, 1)",  # Màu nền tổng thể nhẹ
        dragmode="pan"  # Cho phép kéo (xoay) nhưng tắt zoom
    )

# Thêm màu sắc và kích thước cho các traits
    colors = [ 'blue', 'purple','red', 'orange', '#f1d800','green']
    if len(traits) > len(colors):
        # Extend the colors list by repeating it as needed
        colors = colors * (len(traits) // len(colors) + 1)
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[f'<b style="color:{colors[i]};font-size:16px;">{traits[i]}</b>' for i in range(len(traits))],
                tickmode='array',
            )
        ),
        transition={
        'duration': 1000,
        'easing': 'cubic-in-out'
    }
    )

    # Cập nhật hover label màu theo điểm số
    fig.update_traces(
        hoverlabel=dict(
            font=dict(size=16, color=[get_score_color(score, language) for score in scores]),
            bgcolor='white'  # Mỗi điểm có màu tương ứng với mức độ
        )
    )

    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True)

# ---------------------NHẬN XET---------------------------------
# Hàm lấy nhận xét dựa trên điểm số và trait
def load_financial_traits_descriptions(traits_descriptions_path):
    return pd.read_csv(traits_descriptions_path)

# Đường dẫn tới file CSV chứa mô tả về các đặc điểm tài chính
traits_descriptions_path = 'financial_traits_descriptions.csv'

# Đọc dữ liệu từ file CSV vào DataFrame
descriptions_df = load_financial_traits_descriptions(traits_descriptions_path)

# Thêm tùy chọn chọn ngôn ngữ
language = st.selectbox("Chọn ngôn ngữ / Language settings", ["Tiếng Việt", "English"])

# Hàm để lấy mô tả theo điểm số dựa trên ngôn ngữ
def get_trait_description(trait, score, descriptions_df, language):
    for _, row in descriptions_df.iterrows():
        trait_name = row['Trait']
        if trait_name == trait:
            score_range = row['Score Range']
            min_score, max_score = map(float, score_range.split('-'))
            if min_score <= score <= max_score:
                if language == "Tiếng Việt":
                    return row['Description_vi']
                else:
                    return row['Description_en']
    return "Không có mô tả phù hợp cho trait này." if language == "Tiếng Việt" else "No suitable description for this trait."

# Hàm để lấy nhận xét tổng quan dựa trên top 3 traits
def get_top_traits_description(top_3_traits, descriptions_df, language):
    user_top_traits = '-'.join(top_3_traits).strip().lower().replace('–', '-')   
    print(f"User Traits: {user_top_traits}")  # In ra giá trị của user_top_traits   
    for _, row in descriptions_df.iterrows():
        csv_top_traits = row['Top_3_traits'].strip().lower().replace('–', '-').replace(' ', '') 
        # print(f"CSV Traits: {csv_top_traits}")  # In ra giá trị của csv_top_traits
        if csv_top_traits == user_top_traits:
            if language == "Tiếng Việt":
                return row['top_traits_description_vi']
            else:
                return row['top_traits_description_en']
    
    return "Không có mô tả cho tổ hợp traits này." if language == "Tiếng Việt" else "No description for this combination of traits."

# Hàm để tính top 3 traits dựa trên final_scores
def get_top_3_traits(final_scores):
    # Sắp xếp final_scores theo giá trị (điểm) giảm dần và lấy ra 3 trait đầu tiên
    sorted_traits = sorted(final_scores, key=final_scores.get, reverse=True)
    return sorted_traits[:3]  # Trả về 3 traits có điểm cao nhất
# Hàm để lấy trait cao nhất và thấp nhất

def get_highest_and_lowest_trait(final_scores):
    sorted_traits = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    highest_trait = sorted_traits[0][0]  # Trait có điểm cao nhất
    lowest_trait = sorted_traits[-1][0]  # Trait có điểm thấp nhất
    return highest_trait, lowest_trait

# Hàm để tạo nhận xét dựa trên trait cao nhất và thấp nhất
def get_highest_and_lowest_description(highest_trait, lowest_trait, descriptions_df, language):
    for _, row in descriptions_df.iterrows():
        if row['Highest_Trait'] == highest_trait and row['Lowest_Trait'] == lowest_trait:
            if language == "Tiếng Việt":
                return row['Description_vi_fallback']
            else:
                return row['Description_en_fallback']
    return "Không có mô tả liên quan cho sự kết hợp này." if language == "Tiếng Việt" else "No related description for this combination."


# -----------------HỆ THỐNG ĐIỂM PRODUCT-----------------------------------------------------------------------

# Hàm để đọc điều kiện từ file CSV và chuyển thành hàm lambda
def load_conditions_from_csv(file_path):
    conditions_df = pd.read_csv(file_path)
    conditions_dict = {}
    for _, row in conditions_df.iterrows():
        # Sử dụng `eval` để chuyển chuỗi điều kiện thành hàm lambda thực thi được
        conditions_dict[row['Product']] = eval(f"lambda scores: {row['Condition']}")
    return conditions_dict

# Đọc product_conditions từ file CSV
product_conditions = load_conditions_from_csv('product_conditions.csv')

# Đọc necessity_rules từ file CSV
necessity_rules = load_conditions_from_csv('necessity_rules.csv')

# RandomForest function for model-based eligibility checking
def evaluate_product_with_model(clf, final_scores):
    features = np.array(list(final_scores.values())).reshape(1, -1)
    prediction = clf.predict(features)
    return prediction[0]

# Check eligibility for products
def check_product_eligibility(product, final_scores, clf=None):
    if clf:
        return evaluate_product_with_model(clf, final_scores)
    if product in product_conditions:
        return product_conditions[product](final_scores)
    return True

# Check necessity for products
def evaluate_product_necessity(final_scores, product, clf=None):
    if clf:
        return evaluate_product_with_model(clf, final_scores)
    if product in necessity_rules:
        return necessity_rules[product](final_scores)
    return False

# Calculate product scores using traits and weights
def calculate_product_scores_numpy(final_scores, product_keywords, keyword_to_trait_mapping, weight_factor=1.0):
    trait_names = list(final_scores.keys())
    trait_scores = np.array([final_scores[trait] for trait in trait_names])
    
    product_scores = {}
    for product, keywords in product_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in keyword_to_trait_mapping:
                keyword_trait_weights = np.array([keyword_to_trait_mapping[keyword].get(trait, 0) for trait in trait_names])
                score += weight_factor * np.dot(trait_scores, keyword_trait_weights)
        
        # Sửa lại để trả về một dictionary chứa 'Score', 'Eligible', 'Necessary'
        product_scores[product] = {
            'Score': score,
            'Eligible': check_product_eligibility(product, final_scores),
            'Necessary': evaluate_product_necessity(final_scores, product)
        }
    return product_scores



# Hàm để tìm số cụm tối ưu dựa trên Silhouette Score
def find_optimal_clusters(scores, max_clusters=10):
# DÙng silhouette_scores
    silhouette_scores = []
    for n_clusters in range(3, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scores)
        score = silhouette_score(scores, labels)
        silhouette_scores.append((n_clusters, score))
    
    # Tìm n_clusters có silhouette score cao nhất
    optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]

    print(f"Number of clusters: {optimal_n_clusters}")
    return optimal_n_clusters

# Cập nhật hàm gán nhãn bằng KMeans clustering
def assign_labels_using_kmeans(product_scores, max_clusters=10, random_state=42):
    labeled_scores = {}

    # Kiểm tra 'Product_Score'
    # print(product_scores) 

    # Sử dụng get để đảm bảo có thể lấy được 'Score' hoặc giá trị mặc định là 0 nếu không có
    scores = np.array([result.get('Score', 0) for result in product_scores.values()]).reshape(-1, 1)

    # Tìm số cụm tối ưu dựa trên điểm số
    optimal_n_clusters = find_optimal_clusters(scores, max_clusters)

    # Áp dụng KMeans
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(scores)

    for i, (product, result) in enumerate(product_scores.items()):
        score = result.get('Score', 0)  # Đảm bảo 'Score' luôn tồn tại
        eligible = result.get('Eligible', True)  # Giá trị mặc định là True
        necessary = result.get('Necessary', False)  # Giá trị mặc định là False

       # Gán nhãn dựa trên giá trị của ngôn ngữ
        if language == "Tiếng Việt":
            label_names = ["Rất phù hợp", "Phù hợp", "Ít quan tâm", "Có thể không quan tâm"]
        else:
            label_names = ["Very Suitable", "Suitable", "Little interest", "Might not be interested"]

        label = labels[i]
        label_name = label_names[label] if label < len(label_names) else label_names[-1]

        if eligible and label_name == label_names[-1] and score > 68:
            label_name = label_names[0]  # "Eligible" hoặc "Hợp lệ"

        if necessary:
            label_name = f"Necessary - {label_name}" if language == "English" else f"Cần thiết - {label_name}"

        labeled_scores[product] = {'Score': score, 'Eligible': eligible, 'Necessary': necessary, 'Label': label_name}

    return labeled_scores



# Get final product scores (combines eligibility, necessity, and scoring)
def get_final_product_scores(final_scores, product_keywords, keyword_to_trait_mapping, clf=None, language="Tiếng Việt"):
    product_scores = calculate_product_scores_numpy(final_scores, product_keywords, keyword_to_trait_mapping)

    product_info = {}
    for product, score in product_scores.items():
        is_eligible = check_product_eligibility(product, final_scores, clf)
        is_necessary = evaluate_product_necessity(final_scores, product, clf)
        
        product_info[product] = {
            'Score': score,
            'Eligible': is_eligible,
            'Necessary': is_necessary
        }

    # Điều chỉnh gán nhãn theo ngôn ngữ
    product_info = assign_labels_using_kmeans(product_info, language=language)
    return product_info
# ----------------------------Streamlit UI---------------------------------------------

# Streamlit UI
st.markdown(
    """
    <style>
    .emoji {
        font-size: 50px;
        text-align: center;
    }
    .title {
        font-size: 50px;
        color: #6A0DAD;
        text-align: center;
        display: inline-block;
        background: linear-gradient(90deg, rgba(106,13,173,1) 0%, rgba(163,43,237,1) 50%, rgba(252,176,69,1) 100%);
        -webkit-background-clip: text;
        color: transparent;
        margin: 0 10px;
    }
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        # margin-bottom: 100px; /* Điều chỉnh khoảng cách giữa tiêu đề và tab */
    }

    /* Điều chỉnh khoảng cách tab */
    .stTabs {
        margin-top: -60px; /* Điều chỉnh khoảng cách giữa tiêu đề và tab */
    
    }
    </style>
    <div class="container">
        <div class="emoji">✨</div>
        <div class="title">ASTROTOMI</div>
        <div class="emoji">✨</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>

     /* Tạo lớp phủ nền riêng biệt */
    .bg-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('https://chiemtinhlaso.com/assets/images/hand_bg.png');
        background-size: contain;
        background-position: center;  /* Đảm bảo hình nền căn giữa */
        background-repeat: no-repeat;
        animation: rotate-bg 60s infinite linear;  /* Animation xoay vòng */
        opacity: 0.5;  /* Điều chỉnh độ trong suốt */
    }

    /* Animation xoay vòng cho background */
    @keyframes rotate-bg {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }

    /* Đảm bảo phần nội dung không bị ảnh hưởng */
    [data-testid="stApp"] > div {
        position: relative;
        overflow-y: auto;  /* Cho phép cuộn nội dung */
        height: 100%;  /* Đảm bảo nội dung sử dụng toàn bộ chiều cao */
    }

    /* Đổi nền của sidebar bằng ảnh */
    [data-testid="stSidebar"] {
        background-image: url('https://images.newscientist.com/wp-content/uploads/2023/07/03093742/sei162306009.jpg');
        background-size: cover;  /* Ảnh sẽ bao phủ toàn bộ sidebar */
        background-repeat: no-repeat;
        background-position: relative;
        resize: horizontal;  /* Cho phép kéo ngang */
        # overflow: auto;  /* Đảm bảo nội dung có thể cuộn nếu vượt quá kích thước */
    }

    /* Tạo lớp overlay bán trong suốt */
    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.15); /* Điều chỉnh độ trong suốt tại đây */
        z-index: 0;  /* Đặt lớp overlay phía sau nội dung */
    }

    /* Lật ngược chỉ phần background */
    [data-testid="stSidebar"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: inherit;  /* Kế thừa hình ảnh nền */
        background-size: cover;
        background-position: center; #relative;
        transform: scaleY(-1);  /* Lật ngược hình nền */
        z-index: -1;  /* Đặt hình nền phía sau overlay */
        opacity: 0.7;  /* Đặt độ mờ nếu cần */
    }

    /* Đổi màu chữ của tiêu đề trong sidebar */
    [data-testid="stSidebar"] label {
        color: #f8e4ff;
    }

    [data-testid="stSidebar"] h2 {
        color: #f6d1fb; /* h2 là nhập thông tin... */
    }

    /* Đổi màu chữ cho các phần tử text, button */
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1l02zno, [data-testid="stSidebar"] .css-1vbd788, [data-testid="stSidebar"] .css-1f97m6v {
        color: #0e0834 !important;
    }

    /* Đổi màu cho nút button trong sidebar */
    [data-testid="stSidebar"] button {
        background-color: #601b9f;
        color: white;
    }
 
    /* Đổi màu cho radio buttons */
    [data-testid="stSidebar"] .stRadio label {
        color: #f6d1fb !important;  /*  Màu chữ của AM/PM  */
    }

    /* Đổi màu của placeholder trong input text */
    [data-testid="stSidebar"] input::placeholder {
        color: #0e0834 !important;  /*input text*/
    }
    
    /* Đổi màu chữ cho label của date_input trong sidebar */
    [data-testid="stSidebar"] div.stDateInput label {
        color: #f6d1fb !important;  /* yymmdd */
    }

    /* Đổi màu chữ cho label của number_input trong sidebar */
    [data-testid="stSidebar"] div.stNumberInput label {
        color: #f6d1fb !important;  /* hour minute */ #630c80
    }

    /* Đổi màu chữ trong input */
    input {
        color: f6d1fb !important;
    }

    [data-testid="stSidebar"] {
        width: 400px;  /* Điều chỉnh chiều rộng của sidebar */
    }

    /* Thay đổi màu nền của expander */
    [data-testid="stExpander"] {
        background-color: rgba(240, 242, 246, 0.5); /* Thay đổi thành màu bạn muốn */
    }
    
    /* Thay đổi màu nền của table */
    .custom-table {
        background-color: rgba(240, 242, 246, 0.5);  /* Màu nền của bảng */
        color: black;  /* Màu chữ cho tiêu đề cột */
        border-collapse: collapse;
        width: 99%;
        text-align: left;
    }
    .custom-table th {
        # background-color: rgba(98, 102, 166, 0.1);  /* Màu nền cho tiêu đề cột */
        color: #7f6d84;  /* Màu chữ cho tiêu đề cột */
        text-align: left;
        padding: 4px;
    }
    .custom-table td {
        # background-color: rgba(98, 102, 166, 0.1);  /* Màu nền cho các hàng */
        color: black;  /* Màu chữ cho các hàng */
        padding: 4px;
    }
    # .custom-table tr:nth-child(even) {
    #     background-color: rgba(98, 102, 166, 0.1);  /* Màu nền cho hàng chẵn */
    # }


    </style>
    # <div class="bg-container"></div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header('Nhập thông tin của bạn:'if language == "Tiếng Việt" else "Enter your informations:")

# Dịch văn bản dựa trên ngôn ngữ đã chọn
if language == "Tiếng Việt":
    date_label = 'Ngày sinh (yyyy/mm/dd)'
    hour_label = "Giờ"
    minute_label = "Phút"
    am_pm_label = "AM/PM"
    not_sure_time_msg = "Nếu không rõ giờ sinh, hãy để 11h59p AM."
    birth_place_label = "Nhập tên thành phố:"
    suggestion_label = "Chọn một thành phố từ gợi ý:"
    no_suggestions_msg = "Không tìm thấy thành phố nào phù hợp."
    enter_city_msg = "Nhập tên thành phố để tìm kiếm nơi sinh của bạn."
    text = "Vui lòng nhập ngày tháng năm sinh mà bạn được sinh ra"
else:
    date_label = 'Date of Birth (yyyy/mm/dd)'
    hour_label = "Hour"
    minute_label = "Minute"
    am_pm_label = "AM/PM"
    not_sure_time_msg = "If you're unsure of the birth time, use 11:59 AM."
    birth_place_label = "Enter city name:"
    suggestion_label = "Select a city from suggestions:"
    no_suggestions_msg = "No matching city found."
    enter_city_msg = "Enter city or country name to see suggestions."
    text = "Please enter the date you were born"

# Nhập ngày sinh
birth_date = st.sidebar.date_input(date_label, min_value=datetime(1700, 1, 1), max_value=datetime.today())

st.sidebar.markdown(
    f'<p style="color:white;">{text}</p>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    hour = st.sidebar.number_input(hour_label, min_value=0, max_value=12, value=11)  # Đặt giá trị mặc định là 11
with col2:
    minute = st.sidebar.number_input(minute_label, min_value=0, max_value=59, value=59)  # Đặt giá trị mặc định là 59
with col3:
    am_pm = st.sidebar.radio(am_pm_label, ["AM", "PM"], index=0)  # Chọn AM làm mặc định

# Thêm câu thông báo về giờ sinh không rõ
st.sidebar.markdown(
    f'<p style="color:white;">{not_sure_time_msg}</p>',
    unsafe_allow_html=True
)

# Chuyển đổi sang định dạng 24 giờ
if am_pm == "PM" and hour != 12:
    hour += 12
elif am_pm == "AM" and hour == 12:
    hour = 0

# Nhập địa điểm sinh
birth_place = st.sidebar.text_input(birth_place_label)
if birth_place:
    suggestions = get_city_suggestions(birth_place)
    if suggestions:
        selected_city = st.sidebar.selectbox(suggestion_label, suggestions)
        st.sidebar.markdown(
            f'<p style="color:white;">{f"You have selected: {selected_city}" if language == "English" else f"Bạn đã chọn: {selected_city}"}</p>',
            unsafe_allow_html=True
        )
    else:
        # st.sidebar.write(no_suggestions_msg)
        st.sidebar.markdown(
    f'<p style="color:white;">{no_suggestions_msg}</p>',
    unsafe_allow_html=True
)
else:
    # st.sidebar.write(enter_city_msg)
    st.sidebar.markdown(
    f'<p style="color:white;">{enter_city_msg}</p>',
    unsafe_allow_html=True)



# Khi nhấn nút "Calculate"
if st.sidebar.button("✨Calculate✨"):
    lat, lon, timezone = get_location_and_timezone(birth_place)
    positions = get_planet_positions(birth_date.year, birth_date.month, birth_date.day, hour, minute, lat, lon, timezone)
    ascendant, houses = calculate_houses_and_ascendant(birth_date.year, birth_date.month, birth_date.day, hour, minute, lat, lon, timezone)

    # Tạo danh sách individual_planets từ kết quả positions
    individual_planets = [(planet, get_zodiac_sign_and_degree(degree)[0]) for planet, degree in positions.items()]

    # Hiển thị bảng vị trí các hành tinh với House Cup và House Ruler (bao gồm cả Ascendant)
    df_positions = create_astrology_dataframe(positions, houses, ascendant)
    df_houses = create_house_dataframe(houses, ascendant)

    # Tính toán và hiển thị bảng các góc aspect
    aspects = calculate_aspects(positions)
    aspect_degree_mapping = {
    'Conjunction': 0,
    'Sextile': 60,
    'Square': 90,
    'Trine': 120,
    'Quincunx': 150,
    'Opposition': 180
}
    # Tạo DataFrame cho các góc hợp (aspects)
    df_aspects = create_aspects_dataframe(aspects, positions)


    # Thay đổi cột Degree bằng cách ánh xạ Aspect
    df_aspects['Degree'] = df_aspects['Aspect'].map(aspect_degree_mapping)
    df_aspects['Degree'] = df_aspects['Degree'].apply(lambda x: f"{x}°")
    # Hiển thị bảng đã cập nhật với các cột theo thứ tự mong muốn
    df_aspects = df_aspects[['Planet 1', 'Zodiac Sign 1', 'Aspect', 'Degree', 'Planet 2', 'Zodiac Sign 2']]
   
   
   
    # Tính toán các đặc điểm tài chính
    formatted_aspects = format_aspects({'Aspects': str(aspects)}, individual_planets)
    relevant_planets = [planet for planet, sign in individual_planets]  # Chỉ lấy tên các hành tinh
    extracted_aspects = extract_relevant_aspects(formatted_aspects, relevant_planets)

    final_scores = calculate_financial_traits(individual_planets, formatted_aspects)
    average_scores = {trait: 3.0 for trait in final_scores.keys()}  # Ví dụ tất cả trung bình là 3.0

    final_product_scores = calculate_product_scores_numpy(final_scores, product_keywords, keyword_to_trait_mapping)

    # Sử dụng KMeans clustering để gán nhãn
    labeled_product_scores = assign_labels_using_kmeans(final_product_scores)

    # Nhãn cho các trạng thái sản phẩm
    label_names = {
        "Tiếng Việt": {
            "Very Suitable": "Rất phù hợp",
            "Suitable": "Phù hợp",
            "Little interest": "Ít quan tâm",
            "Might not be interested": "Có thể không quan tâm"
        },
        "English": {
            "Very Suitable": "Very Suitable",
            "Suitable": "Suitable",
            "Little interest": "Little interest",
            "Might not be interested": "Might not be interested"
        }
    }

    # Chọn ngôn ngữ hiện tại
    current_language = language  # Hoặc "Tiếng Việt" hoặc "English"

    # Phân loại sản phẩm theo Eligible và Necessary
    eligible_products = [
        (product, result['Score'], result['Label'].replace("Cần thiết - ", "").replace("Necessary - ", ""))
        for product, result in labeled_product_scores.items() 
    ]

    necessary_products = [
        (product, result['Score'], label_names[current_language].get(result['Label'], result['Label']))
        for product, result in labeled_product_scores.items() if result['Necessary']
    ]
    

    # Tạo DataFrame cho Eligible và Necessary Products
    eligible_df = pd.DataFrame(eligible_products, columns=['Product', 'Score', 'Label'])
    necessary_df = pd.DataFrame(necessary_products, columns=['Product', 'Score', 'Label'])

    # Mức độ ưu tiên của các nhãn
    priority = {
        "Rất phù hợp": 1,
        "Phù hợp": 2,
        "Ít quan tâm": 3,
        "Có thể không quan tâm": 4
    } if current_language == "Tiếng Việt" else {
        "Very Suitable": 1,
        "Suitable": 2,
        "Little interest": 3,
        "Might not be interested": 4
    }

    def custom_sort(row):
        label = row['Label']
        if current_language == "Tiếng Việt":
            if "Cần thiết" in label:
                label_without_necessity = label.replace("Cần thiết - ", "")
                return (2, priority.get(label_without_necessity, 5))  
            else:
                return (1, priority.get(label, 5))
        elif current_language == "English":
            if "Necessary" in label:
                label_without_necessity = label.replace("Necessary - ", "")
                return (2, priority.get(label_without_necessity, 5))
            else:
                return (1, priority.get(label, 5))


    # Sắp xếp theo thứ tự ưu tiên và hiển thị cho các sản phẩm
    eligible_df['Priority'] = eligible_df['Label'].map(priority)  
    eligible_df = eligible_df.sort_values(by=['Priority', 'Score'], ascending=[True, False]).drop(columns='Priority').reset_index(drop=True)

    necessary_df['Sort_Order'] = necessary_df.apply(custom_sort, axis=1)
    necessary_df = necessary_df.sort_values(by=['Sort_Order', 'Score', 'Label'], ascending=[True, False, True]).drop(columns=['Sort_Order']).reset_index(drop=True)

    # Tạo DataFrame cho tất cả các sản phẩm


    # Sắp xếp và hiển thị cho tất cả các sản phẩm
    all_products = [
        (product, result['Score'], label_names[current_language].get(result['Label'], result['Label']))
        for product, result in labeled_product_scores.items()
    ]
    all_products_df = pd.DataFrame(all_products, columns=['Product', 'Score', 'Label'])
    all_products_df['Priority'] = all_products_df['Label'].apply(lambda x: 0 if x.startswith("Necessary - ") or x.startswith("Cần thiết - ") else 1)

    all_products_df = all_products_df.sort_values(by=['Priority', 'Score'], ascending=[True, False]).drop(columns='Priority').reset_index(drop=True)
    
    
    # Tạo các tab cho các phần khác nhau
    tab1, tab2, tab3 = st.tabs(["Astrology", "Financial Traits", "Product Recommendations"])

    # Tab 1: Hiển thị vị trí các hành tinh
    with tab1:
        st.write("### Planetary Positions:")
        st.dataframe(df_positions)
        st.write("### House Information:")
        st.dataframe(df_houses)
        st.write("### Planetary Aspects:")
        st.dataframe(df_aspects)
        

    # Tab 2: Hiển thị Radar Chart cho các đặc điểm tài chính
    with tab2:
        if language == "Tiếng Việt":
            st.write("### Biểu đồ dựa trên hành vi tài chính của bạn:")
        else:
            st.write("### Financial Traits Radar Chart:")

        plot_radar_chart(final_scores, average_scores)

        # Generate nhận xét từ tất cả các traits
        if language == "Tiếng Việt":
            st.write("### Mô tả về dựa trên tính cách:")
        else:
            st.write("### Trait Descriptions:")
        
        # Sắp xếp các traits theo điểm từ cao đến thấp
        sorted_traits = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Thêm phần mở đầu
        # st.write("Bạn có các đặc điểm tài chính nổi bật như sau:" if language == "Tiếng Việt" else "You have the following prominent financial traits:")

        # Tạo đoạn văn bản mô tả tất cả các traits với số thứ tự
        financial_traits_text = ""
        trait_colors = {
            "Adventurous": "blue",
            "Convenience": "purple",
            "Impulsive": "red",
            "Conservative": "orange",
            "Caution": "#f1d800",
            "Analytical": "green"
        }

        for index, (trait, score) in enumerate(sorted_traits):
            if index == 0:
                note = " - cao nhất" if language == "Tiếng Việt" else " - highest"
            elif index == len(sorted_traits) - 1:
                note = " - thấp nhất" if language == "Tiếng Việt" else " - lowest"
            else:
                note = ""

            # Lấy màu sắc dựa trên trait
            color = trait_colors.get(trait.capitalize(), "black")  # Mặc định là "black" nếu không có trong trait_colors

            description = get_trait_description(trait.capitalize(), score, descriptions_df, language)
            
            # Sử dụng HTML để thay đổi màu sắc của trait
            financial_traits_text += f"{index + 1}. <b style='color:{color};'>{trait.capitalize()}</b> **({score:.3f}{note}):**\n\n {description}\n\n"

        # Hiển thị toàn bộ đoạn văn bản
        with st.expander("**Chi tiết các đặc điểm tài chính**" if language == "Tiếng Việt" else "**Financial traits**"):
            st.markdown(financial_traits_text, unsafe_allow_html=True)

        # Hiển thị mô tả cho top 3 traits
        top_3_traits = get_top_3_traits(final_scores)
        top_traits_description = get_top_traits_description(top_3_traits, descriptions_df, language)

        # Hiển thị tiêu đề nhận xét tổng quát dựa trên ngôn ngữ được chọn
        st.write("### Nhận xét về hành vi tài chính:" if language == "Tiếng Việt" else "### Financial behavior insights:")

        # Sử dụng expander để ẩn/hiện phần nhận xét chi tiết
        with st.expander("**Nhận xét chi tiết**" if language == "Tiếng Việt" else "**Detailed financial behavior insights:**"):
            st.write(top_traits_description)

        # # Nhận xét về trait cao nhất và thấp nhất
        # highest_trait, lowest_trait = get_highest_and_lowest_trait(final_scores)
        # highest_lowest_description = get_highest_and_lowest_description(highest_trait, lowest_trait, descriptions_df, language)

        # st.write("### Nhận xét về sự kết hợp giữa trait cao nhất và thấp nhất:" if language == "Tiếng Việt" else "### Insights on the combination of highest and lowest trait:")
        
        # # Sử dụng expander để hiển thị nhận xét
        # with st.expander("**Nhận xét chi tiết về sự kết hợp**" if language == "Tiếng Việt" else "**Detailed insights on the trait combination:**"):
        #     st.write(highest_lowest_description)


    # Tab 4: Hiển thị sản phẩm được gợi ý (Eligible và Necessary Products)
    with tab3:
        if language == "Tiếng Việt":
            st.subheader("Bạn sẽ thích: ")
            # with st.expander("Bạn sẽ thích: "):
                # st.table(eligible_df)
            st.markdown(eligible_df.to_html(classes='custom-table'), unsafe_allow_html=True)

            st.subheader("Bạn sẽ cần: ")
            # st.table(necessary_df)
            st.markdown(necessary_df.to_html(classes='custom-table'), unsafe_allow_html=True)

            st.subheader("Tất cả sản phẩm")
            # st.table(all_products_df)
            st.markdown(all_products_df.to_html(classes='custom-table'), unsafe_allow_html=True)
        else:
            st.subheader("Your matchs:")
            # st.table(eligible_df)
            st.markdown(eligible_df.to_html(classes='custom-table'), unsafe_allow_html=True)

            st.subheader("You will need: ")
            # st.table(necessary_df)
            st.markdown(necessary_df.to_html(classes='custom-table'), unsafe_allow_html=True)

            st.subheader("All Products")
            # st.table(all_products_df)
            st.markdown(all_products_df.to_html(classes='custom-table'), unsafe_allow_html=True)
