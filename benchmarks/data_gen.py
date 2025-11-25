import io
import csv
from typing import Dict, Any, List

def generate_neutral_input(dtype: str, count: int = 10) -> List[Dict[str, Any]]:
    """
    Generates data in a 'Neutral' format (CSV or Text) to be used as the prompt Input.
    Returns: List of {'prompt_text': str, 'ground_truth': List[Dict], 'format': str}
    """
    results = []
    
    if dtype == "Tabular":
        # CSV Format - 10 variations
        # Standardized Products Dataset for TabularFlat
        # Schema: id (int), name (str), cat (str), price (float), stock (bool)
        base_products = [
            {"id": 101, "name": "Super Widget", "cat": "Hardware", "price": 99.50, "stock": True},
            {"id": 102, "name": "Mega Gadget", "cat": "Hardware", "price": 149.00, "stock": False},
            {"id": 103, "name": "Cable Pack", "cat": "Accessory", "price": 19.99, "stock": True},
            {"id": 104, "name": "Ultra Monitor", "cat": "Hardware", "price": 299.99, "stock": True},
            {"id": 105, "name": "Mouse Pad", "cat": "Accessory", "price": 9.99, "stock": True},
            {"id": 106, "name": "Keyboard", "cat": "Peripherals", "price": 49.50, "stock": False},
            {"id": 107, "name": "USB Hub", "cat": "Accessory", "price": 24.99, "stock": True},
            {"id": 108, "name": "Webcam", "cat": "Peripherals", "price": 79.00, "stock": True},
            {"id": 109, "name": "Headset", "cat": "Audio", "price": 59.99, "stock": False},
            {"id": 110, "name": "Microphone", "cat": "Audio", "price": 89.99, "stock": True},
        ]
        
        datasets = []
        # Generate 10 variations by slicing/shuffling or just repeating for now
        # To keep it simple and consistent with the model, we'll just create chunks of the base_products
        # or slightly modify them.
        import copy
        for i in range(10):
            # Create a variation
            variation = []
            for item in base_products:
                new_item = copy.deepcopy(item)
                new_item['id'] += (i * 100) # Offset ID
                new_item['price'] += (i * 1.5) # Vary price
                variation.append(new_item)
            # Take a subset of 3-5 items
            import random
            # Use fixed seed for reproducibility if needed, but here we just want valid data
            subset_size = 3 + (i % 3)
            datasets.append(variation[:subset_size])
        
        for data in datasets[:count]:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            results.append({"prompt_text": output.getvalue(), "ground_truth": data, "format": "CSV"})
    
    elif dtype == "Text-Heavy":
        # Bulleted List Format - 10 variations
        datasets = [
            # Product Reviews
            [
                {"id": "R1", "user": "alice", "text": "The build quality is fantastic, solid aluminum feel. However, the battery life is mediocre at best."},
                {"id": "R2", "user": "bob", "text": "Shipping was fast. The item arrived well packaged. Setup was a breeze, took less than 5 minutes."},
                {"id": "R3", "user": "charlie", "text": "Terrible software experience. The app crashes constantly and the UI looks like it's from 2010. Avoid."},
            ],
            # Customer feedback
            [
                {"id": "F1", "user": "dana", "text": "Amazing customer service! The support team resolved my issue within 24 hours. Highly recommended."},
                {"id": "F2", "user": "eve", "text": "Product works as advertised but the price is too high for what you get. Expected more features."},
                {"id": "F3", "user": "frank", "text": "Defective unit received. Contacted support multiple times with no response. Very disappointed."},
            ],
            # Meeting notes
            [
                {"id": "M1", "user": "alice", "text": "Discussed Q4 roadmap. Team agreed to prioritize mobile app development over desktop features."},
                {"id": "M2", "user": "bob", "text": "Budget review meeting. Approved $50K for marketing campaign. Need approval from CFO for additional funds."},
                {"id": "M3", "user": "carol", "text": "Sprint planning complete. 15 story points committed. Focus on bug fixes this iteration."},
            ],
            # Support tickets
            [
                {"id": "T1", "user": "user123", "text": "Cannot login to my account. Password reset link not working. Please help urgently!"},
                {"id": "T2", "user": "user456", "text": "Billing issue - charged twice for same transaction. Need refund processed ASAP."},
                {"id": "T3", "user": "user789", "text": "Feature request: Add dark mode to mobile app. Many users are asking for this."},
            ],
            # Blog comments
            [
                {"id": "C1", "user": "techie", "text": "Great article! The examples were clear and easy to follow. Looking forward to more content like this."},
                {"id": "C2", "user": "learner", "text": "Could you provide more details on the implementation? The code snippet is missing some imports."},
                {"id": "C3", "user": "critic", "text": "Disagree with your conclusion. There are better approaches that you didn't mention here."},
            ],
            # Survey responses
            [
                {"id": "S1", "user": "resp1", "text": "Very satisfied with the service. Would definitely recommend to friends and family."},
                {"id": "S2", "user": "resp2", "text": "Neutral experience. Nothing exceptional but no major complaints either. Average service."},
                {"id": "S3", "user": "resp3", "text": "Extremely disappointed. Will not be using this service again. Looking for alternatives."},
            ],
            # Social media posts
            [
                {"id": "P1", "user": "influencer", "text": "Just tried the new feature and it's amazing! Game changer for productivity. #tech"},
                {"id": "P2", "user": "casual", "text": "Anyone else having issues with the latest update? App keeps freezing on my device."},
                {"id": "P3", "user": "fan", "text": "Been using this for 3 years now. Still the best tool in its category. Worth every penny!"},
            ],
            # Email summaries
            [
                {"id": "E1", "user": "john@ex.com", "text": "Please review the attached proposal by EOD. Meeting scheduled for tomorrow at 2 PM."},
                {"id": "E2", "user": "jane@ex.com", "text": "Reminder: Annual team building event next Friday. RSVP required by Wednesday."},
                {"id": "E3", "user": "bob@ex.com", "text": "System maintenance window scheduled for this weekend. Services will be down 2-4 hours."},
            ],
            # News headlines
            [
                {"id": "N1", "user": "reporter", "text": "Breaking: Tech company announces major layoffs affecting 10% of workforce amid restructuring."},
                {"id": "N2", "user": "editor", "text": "Market update: Stocks reach all-time high following positive economic indicators and strong earnings."},
                {"id": "N3", "user": "analyst", "text": "Industry report: AI adoption increasing rapidly across sectors, transforming business operations."},
            ],
            # App store reviews
            [
                {"id": "A1", "user": "happyuser", "text": "Love this app! So intuitive and easy to use. Best $5 I've ever spent. Five stars!"},
                {"id": "A2", "user": "frustrated", "text": "Used to be great but recent updates have made it buggy. Please fix the performance issues."},
                {"id": "A3", "user": "newbie", "text": "Just downloaded. Looks promising but still learning the features. Tutorial was helpful."},
            ],
        ]
        
        for data in datasets[:count]:
            txt = ""
            for d in data:
                txt += f"- Review {d['id']} by {d['user']}: \"{d['text']}\"\n"
            results.append({"prompt_text": txt, "ground_truth": data, "format": "Text List"})
    
    elif dtype == "Nested":
        # Structured Text Block - 10 variations
        # Standardized User Profiles for NestedFlat
        # Schema: uid (str), profile_name (str), profile_city (str), tags (List[str])
        # Input data is nested: {"uid": "...", "profile": {"name": "...", "city": "..."}, "tags": [...]}
        base_profiles = [
            {"uid": "u1", "profile": {"name": "John", "city": "NY"}, "tags": ["admin", "editor"]},
            {"uid": "u2", "profile": {"name": "Jane", "city": "LA"}, "tags": ["viewer"]},
            {"uid": "u3", "profile": {"name": "Dave", "city": "SF"}, "tags": ["viewer", "guest"]},
            {"uid": "u4", "profile": {"name": "Alice", "city": "Chicago"}, "tags": ["admin"]},
            {"uid": "u5", "profile": {"name": "Bob", "city": "Seattle"}, "tags": ["editor", "contributor"]},
            {"uid": "u6", "profile": {"name": "Carol", "city": "Austin"}, "tags": ["viewer"]},
            {"uid": "u7", "profile": {"name": "Eve", "city": "Boston"}, "tags": ["guest"]},
            {"uid": "u8", "profile": {"name": "Frank", "city": "Denver"}, "tags": ["admin", "viewer"]},
            {"uid": "u9", "profile": {"name": "Grace", "city": "Miami"}, "tags": ["editor"]},
            {"uid": "u10", "profile": {"name": "Heidi", "city": "Portland"}, "tags": ["contributor"]},
        ]

        datasets = []
        import copy
        for i in range(10):
            variation = []
            for item in base_profiles:
                new_item = copy.deepcopy(item)
                new_item['uid'] = f"u{i}_{item['uid']}"
                variation.append(new_item)
            subset_size = 3 + (i % 3)
            datasets.append(variation[:subset_size])
        
        for data in datasets[:count]:
            txt = ""
            for d in data:
                # Get first key as ID, second as nested dict, third as list
                keys = list(d.keys())
                id_key = keys[0]
                nested_key = keys[1]
                list_key = keys[2]
                
                nested_data = d[nested_key]
                nested_str = ", ".join([f"{k}={v}" for k, v in nested_data.items()])
                list_str = ",".join(d[list_key])
                
                txt += f"{id_key}={d[id_key]}: {nested_str}. {list_key}={list_str}\n"
            results.append({"prompt_text": txt, "ground_truth": data, "format": "Structured Text"})
    
    return results
