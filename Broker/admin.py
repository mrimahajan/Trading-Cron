from django.contrib import admin
from Broker.models import *

# Register your models here.
admin.site.register(Stock)
admin.site.register(Stock_Trade)
admin.site.register(Stock_Portfolio)
admin.site.register(Stock_Profit_Loss)
admin.site.register(Exchange)
