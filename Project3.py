import pandas as pd
import numpy as np
import re
from word2number import w2n
from rapidfuzz import process, fuzz
import ast

students = pd.read_csv("super_dirty_students.csv")

# Name based
def name_claen(name):
    if pd.isna(name):
        return np.nan
    
    namee = str(name).strip()

    return namee

students["name"] = students["name"].apply(name_claen)

# Age based

def clean_age(age):
    # Blank
    if pd.isna(age):
        return np.nan
    
    val = str(age).lower().strip()

    # Didgits (18 (years))
    digits =  re.findall(r"\d+", val)
    if digits:
        return int(digits[0])
    
    # Twenty = 20
    try:
        return w2n.word_to_num(val)
    except:
        return np.nan
    
students["age"] = students["age"].apply(clean_age).astype("Int64")

# Gender based

# True values
validgens = ["female", "male"]

def gender_cleaning(gen):
    #  Blank
    if pd.isna(gen):
        return np.nan

    val_g = str(gen).strip().lower()

    # "fe" or "me" can't guess
    if len(val_g) < 3:
        return np.nan
    
    # "fmale" = female/ "male" = "male"
    result = process.extractOne(
        val_g,
        validgens,
        scorer=fuzz.ratio
    )

    if result is None:
        return np.nan
    
    match, score, _ = result

    # Treshold
    if score >= 70:
        return match.capitalize()
    else:
        return np.nan
    
    
students["gender"] = students["gender"].apply(gender_cleaning)

# Score based

def clean_score(score):
    # Blank
    if pd.isna(score):
        return np.nan
    
    val_s = str(score).strip().lower()
    # Didgit
    if val_s.isdigit():
        return float(val_s)
    
    # Ninety = 90
    try:
        return w2n.word_to_num(val_s)
    except:
        return np.nan

students["score"] = students["score"].apply(clean_score).astype('Float64')

# City based

def city_clean(city):
    if pd.isna(city):
        return np.nan
    
    cityy = str(city).strip()

    return cityy

students["city"] = students["city"].apply(city_clean)

# Phone based (extension)


def phone_clean(phone):
    # Blank
    if pd.isna(phone):
        return np.nan, np.nan
    
    # Extension
    match_ext = re.search(r"[xX](\d+)", phone)
    extension = match_ext.group(1) if match_ext else np.nan

    # Extracting Digits
    digits = re.sub(r"\D", "", phone)

    # Formatting Phone number
    if len(digits) >= 10:
        main_part = digits[-10:]
        formatt = f"+1-{main_part[0:3]}-{main_part[3:6]}-{main_part[6:10]}"
    else:
        formatt = np.nan

    return formatt, extension


students[["phone", "extension"]] = students["phone"].apply(lambda x: pd.Series(phone_clean(x)))

# Email based

def email_clean(email):
    # Blank
    if pd.isna(email):
        return np.nan
    
    email_s = str(email).strip().lower()

    # @@ -> @
    if email_s.count("@") > 1:
        email_s = email_s.replace("@@", "@")

    # .. = invalid email
    if ".." in email_s:
        return np.nan
    
    # Valid email, regex
    patternn = r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$"

    if re.match(patternn, email_s):
        return email_s
    else:
        return np.nan

students["email"] = students["email"].apply(email_clean)

# Dropping rows that have not name or phone or email

students = students.dropna(
    subset=["name", "phone", "email"], 
    how="all")


# Join Date based

def join_date_clean(date_join):

    # Blank 
    if pd.isna(date_join):
        return pd.NaT
    
    dj = str(date_join).strip()

    # Convert Unix timestamp to datetime
    if dj.isdigit():
        try:
            if len(dj) == 10:
                return pd.to_datetime(int(dj), unit="s")
            elif len(dj) == 13:
                return pd.to_datetime(int(dj), unit="ms")
        except:
            return pd.NaT

    # Convert dj to datetime
    try:
        return pd.to_datetime(dj, dayfirst=False, errors="coerce")
    except:
        return pd.NaT
    
students["date_of_join"] = students["date_of_join"].apply(join_date_clean)

# Course based

def text_normalization(text):
    txt = str(text).strip().lower()
    txt = re.sub(r"[^a-z]", "", txt)
    return txt

students["course"] = students["course"].apply(text_normalization)

map_course = {
    'datascience': 'Data Science',
    'datasciens': 'Data Science',        
    'datasciense': 'Data Science',
    'ds': 'Data Science',
    'python': 'Python',
    'pythno': 'Python',
    'pyhton': 'Python'
}

students["course"] = students["course"].map(map_course)

# Attendance based

def attendance_clean(perc):
    # Blank
    if pd.isna(perc):
        return np.nan
    
    p = str(perc).strip().replace("%", "")

    # Converting to float
    try:
        percentage = float(p)
    except ValueError:
        return np.nan
    
    # Percentage range

    if percentage < 0:
        return 0.0
    if percentage > 100:
        return 100.0

    return percentage

students["attendance"] = students["attendance"].apply(attendance_clean)

# Status based

def status_clean(status):
    if pd.isna(status):
        return np.nan
    
    statuss = str(status).strip().capitalize()

    return statuss

students["status"] = students["status"].apply(status_clean)

# Gpa based

def clean_gpa(gpa):
    if pd.isna(gpa):
        return np.nan
    
    cleangpa = str(gpa).strip().lower().replace(",", ".")

    try:
        gpaa = float(cleangpa)
    except:
        try:
            gpaa = w2n.word_to_num(cleangpa)
        except:
            return np.nan
    
    if gpaa < 0:
        return np.nan
    if gpaa > 5:
        return np.nan
    
    return float(gpaa)

students["gpa"] = students["gpa"].apply(clean_gpa)

# Remark based

def clean_remark(remark):
    if pd.isna(remark):
        return np.nan
    
    remarkk = str(remark).strip().capitalize()

    return remarkk

students["remarks"] = students["remarks"].apply(clean_remark)


# Money Spent based

def clean_money_spent(money_spent):
    if pd.isna(money_spent):
        return np.nan
    
    s = str(money_spent).replace(",", ".")  

    match = re.search(r"\d+(?:\.\d+)?", s) 

    return float(match.group()) if match else np.nan

students["money_spent"] = students["money_spent"].apply(clean_money_spent)

# Event time based

# Same as date_of_join

def cleac_event_time(event_time):

    # Blank 
    if pd.isna(event_time):
        return pd.NaT
    
    et = str(event_time).strip()

    # Convert Unix timestamp to datetime
    if et.isdigit():
        try:
            if len(et) == 10:
                return pd.to_datetime(int(et), unit="s")
            elif len(et) == 13:
                return pd.to_datetime(int(et), unit="ms")
        except:
            return pd.NaT

    # Convert et to datetime
    try:
        return pd.to_datetime(et, dayfirst=False, errors="coerce")
    except:
        return pd.NaT

students["event_time"] = students["event_time"].apply(cleac_event_time)

# Address based

def address_clean(adr_raw):

    # Blank
    if pd.isna(adr_raw):
        return pd.Series([np.nan, np.nan, np.nan], index=["address", "city", "postal"])
    
    # Converting to Text and removing blanks

    text = str(adr_raw).strip()

    # "BROKEN,...." = no address
    if "BROKEN" in text.upper():
        return pd.Series([np.nan, np.nan, np.nan], index=["address", "city", "postal"])
    
    # Postal 
    # default NaN
    postal = np.nan

    # Main
    pos_uz = re.search(r"\bUZ[\s,]*(\d{5,6})\b", text)

    if pos_uz:
        postal = pos_uz.group(1)

    elif re.search(r",\s*\d{5}$", text):
        postal = re.search(r"(\d{5})$", text).group(1)

    
    # City
    # default NaN
    city = np.nan

    # Main
    if re.search(r"\bTashkent\b", text):
        city = "Tashkent"

    else:
        cityy = re.search(r",\s*([A-Za-z\s]+)$", text)
        if cityy:
            city = cityy.group(1).strip()

    # Extarcting only address

    address = text
    # UZ + postal ext
    address = re.sub(r",?\s*UZ[\s,]*\d{5,6}", "", address)
    # US ext
    address = re.sub(r",\s*\d{5}$", "", address)
    # City name ext
    if city:
        address = re.sub(rf"\b{re.escape(city)}\b", "", address)

    # comma blank, blank comma
    address = re.sub(r",\s*,+", ", ", address)
    address = re.sub(r"\s{2,}", " ", address)
    
    address = address.strip(" ,")

    return pd.Series([address, city, postal], index=["address", "city", "postal"])

students[["address", "add_city", "postal"]] = students["address_raw"].apply(address_clean)

# Profile Json based

# Parsing (with Regex)
def safe_parse(val):
    # INVALID... = no data
    if pd.isna(val) or val == "INVALID_JSON_DATA":
        return {}
    if isinstance(val, dict):
        return val
    
    s = str(val).strip()
    # Adding quotes to keys: {hobbies: -> {'hobbies':
    s = re.sub(r'(\w+)\s*:', r"'\1':", s) 
    
    try:
        return ast.literal_eval(s)
    except:
        # If the parsing still errors, just pull out the hobbies (fallback)
        hobbies_match = re.search(r"hobbies':\s*\[(.*?)\]", s)
        if hobbies_match:
            h_list = hobbies_match.group(1).replace("'", "").split(", ")
            return {"hobbies": [h.strip() for h in h_list]}
        return {}

# function for converting list to str
def safe_list_join(val):
    if isinstance(val, list) and len(val) > 0:
        return ", ".join(map(str, val))
    return np.nan

# function for splitting devices
def split_devices(devices):
    result = {
        "laptop_brand": np.nan, "laptop_year": np.nan,
        "phone_brand": np.nan, "phone_year": np.nan
    }
    if not isinstance(devices, list):
        return result
    for d in devices:
        if not isinstance(d, dict): continue
        dtype = d.get("type")
        if dtype == "laptop":
            result["laptop_brand"] = d.get("brand")
            result["laptop_year"] = d.get("year")
        elif dtype == "phone":
            result["phone_brand"] = d.get("brand")
            result["phone_year"] = d.get("year")
    return result


# Parsing
students["profile_parsed"] = students["profile_json"].apply(safe_parse)

# Normalizing /column names
flat_df = pd.json_normalize(students["profile_parsed"], sep="_")

# opening list for hobbies and skills_soft 
if "hobbies" in flat_df.columns:
    flat_df["hobbies"] = flat_df["hobbies"].apply(safe_list_join)

if "skills_soft" in flat_df.columns:
    flat_df["skills_soft"] = flat_df["skills_soft"].apply(safe_list_join)

# creating columns for devices
device_df = pd.DataFrame(
    students["profile_parsed"].apply(lambda x: x.get("devices")).apply(split_devices).tolist()
)

# collecting finel DataFrame

# delete profile_json and profile_parsed 
students_cleaned = students.drop(columns=["profile_json", "profile_parsed"]).reset_index(drop=True)
flat_df = flat_df.reset_index(drop=True)
device_df = device_df.reset_index(drop=True)

# if devices json in flat_df drop it for no duplicate extra columns
if "devices" in flat_df.columns:
    flat_df = flat_df.drop(columns=["devices"])

students = pd.concat([students_cleaned, flat_df, device_df], axis=1)

# changing some dtypes 

students["family_siblings"] = students["family_siblings"].astype('Int64')
students["laptop_year"] = students["laptop_year"].astype('Int64')
students["phone_year"] = students["phone_year"].astype('Int64')


# information
print(students.info())


# creating .csv in correct order of columns
students[["student_id", 
          "name", 
          "age",
          "gender", 
          "score", 
          "phone", 
          "extension", 
          "email", 
          "date_of_join", 
          "course", 
          "attendance", 
          "status", 
          "gpa", 
          "remarks",
          "money_spent",
          "event_time",
          "address",
          "city",
          "postal",
          "hobbies",
          "skills_tech_python",
          "skills_tech_excel",
          "skills_tech_sql",
          "skills_soft",
          "family_siblings",
          "family_income_father",
          "family_income_mother",
          "laptop_brand",
          "laptop_year",
          "phone_brand",
          "phone_year"]].to_csv("super_dirty_students_cleaned.csv", index=False)





