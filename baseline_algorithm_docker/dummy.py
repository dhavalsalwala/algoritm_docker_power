import gcapi
import os
from pathlib import Path


token = '53233e141f46901fbd55ca5cb2a66a906a51b4ae69dd918c0bbd67caef1c45f7'
archive_slug = 'shifts-power-estimation-phase-1'

client = gcapi.Client(token=token)

archive = client.archives.detail(slug=archive_slug)
archive_url = archive["api_url"]
print(archive_url)
# Archive items link together related files/image in an archive (eg those that belong to the same case), create a new one here
ai = client.archive_items.create(archive=archive_url, values=[])
print(ai["pk"])
print(ai)
# Upload the two image files to the newly created archive item, each with their own interface (in this case a ct-image and a generic-overlay)
ct= Path("/home/ubuntu/shifts_challenge/notebooks/final_datasets/phase_1/private_data/merchant-vessel-features.json")
client.update_archive_item(archive_item_pk=ai["pk"],
                           values={"merchant-vessel-features": [ct]})