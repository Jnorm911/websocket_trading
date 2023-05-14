import csv
import datetime
import os


def convert_to_unix_time(date_str, time_str):
    datetime_str = date_str + " " + time_str
    dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    unix_time = int(dt.timestamp())
    return unix_time


input_directory = "data/kc/btc/heiken_ashi/raw"
output_directory = "data/kc/btc/heiken_ashi/converted"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for x in range(1, 61):
    input_file = os.path.join(input_directory, f"kc_btc_{x}min_ha.csv")
    output_file = os.path.join(output_directory, f"kc_btc_{x}min_ha.csv")

    with open(input_file, "r") as file:
        reader = csv.reader(file)
        headers = next(reader)
        date_index = headers.index("date")
        time_index = headers.index("time")
        headers.remove("date")  # Remove 'date' column

        rows = []
        for row in reader:
            date = row[date_index]
            time = row[time_index]
            unix_time = convert_to_unix_time(date, time)
            row[time_index] = str(unix_time)  # Replace 'time' value with Unix time
            row.pop(date_index)  # Remove the date value from the row
            rows.append(row)

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Conversion completed for {input_file}. Output file: {output_file}")
