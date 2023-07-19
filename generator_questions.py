import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")  
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

sentence = "Phạm vi điều chỉnh\n\nLuật này quy định về hoạt động bảo vệ môi trường; quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường.."
# text =  "generator questions : " + sentence + " </s>"
input_text = "answer: %s  context: %s </s>" % ("Luật bảo vệ môi trường;", sentence)
encoding = tokenizer(input_text, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    early_stopping=True
)
for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(line)
    

from transformers import MT5ForConditionalGeneration, AutoTokenizer
import torch

model = MT5ForConditionalGeneration.from_pretrained(
    "noah-ai/mt5-base-question-generation-vi")
tokenizer = AutoTokenizer.from_pretrained(
    "noah-ai/mt5-base-question-generation-vi")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Content used to create a set of questions
context = '''Thành phố Hồ Chí Minh (còn gọi là Sài Gòn) tên gọi cũ trước 1975 là Sài Gòn hay Sài Gòn-Gia Định là thành phố lớn nhất ở Việt Nam về dân số và quy mô đô thị hóa. Đây còn là trung tâm kinh tế, chính trị, văn hóa và giáo dục tại Việt Nam. Thành phố Hồ Chí Minh là thành phố trực thuộc trung ương thuộc loại đô thị đặc biệt của Việt Nam cùng với thủ đô Hà Nội.Nằm trong vùng chuyển tiếp giữa Đông Nam Bộ và Tây Nam Bộ, thành phố này hiện có 16 quận, 1 thành phố và 5 huyện, tổng diện tích 2.061 km². Theo kết quả điều tra dân số chính thức vào thời điểm ngày một tháng 4 năm 2009 thì dân số thành phố là 7.162.864 người (chiếm 8,34% dân số Việt Nam), mật độ dân số trung bình 3.419 người/km². Đến năm 2019, dân số thành phố tăng lên 8.993.082 người và cũng là nơi có mật độ dân số cao nhất Việt Nam. Tuy nhiên, nếu tính những người cư trú không đăng ký hộ khẩu thì dân số thực tế của thành phố này năm 2018 là gần 14 triệu người.'''

encoding = tokenizer.encode_plus(context, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(
    device), encoding["attention_mask"].to(device)

output = model.generate(input_ids=input_ids,
                        attention_mask=attention_masks, max_length=256)

question = tokenizer.decode(
    output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(question)


