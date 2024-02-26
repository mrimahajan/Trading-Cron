from django.db import models
from django.contrib.auth.models import User

Cap_Choices = (("Large","Large"),("Mid","Mid"),("Small","Small"))

class Stock(models.Model):
    Sector = models.CharField(max_length=50)
    Cap = models.CharField(max_length=10, choices=Cap_Choices)  # Adjusted max_length
    Company = models.CharField(max_length=100)  # Adjusted max_length
    Symbol = models.CharField(max_length=16, primary_key=True)
    CLS_Price = models.FloatField(null=True)
    EOD_Price = models.FloatField(null=True)  # Adjusted field type
    Expected_Price = models.FloatField(null=True)  # Adjusted field type
    net_return = models.FloatField(null=True)  
    risk = models.FloatField(null=True)
    probability = models.FloatField(null=True)
    market_contri_reg = models.FloatField(null=True)
    momentum_contri_reg = models.FloatField(null=True)
    mean_reversion_contri_reg = models.FloatField(null=True)
    voltality_contri_reg = models.FloatField(null=True)
    volume_contri_reg = models.FloatField(null=True)
    market_contri_class = models.FloatField(null=True)
    momentum_contri_class = models.FloatField(null=True)
    mean_reversion_contri_class = models.FloatField(null=True)
    voltality_contri_class = models.FloatField(null=True)
    volume_contri_class = models.FloatField(null=True)

    def __str__(self):
        return str(self.Symbol)
    
    # def save(self,*args,**kwargs):
    #     if self.CLS_Price is not None:
    #         self.CLS_Price += 0
    #     else:
    #         self.CLS_Price = 0
    #     super().save(*args,**kwargs)

Trade_Choices = (("Buy","Buy"),("Sell","Sell"),)

class Stock_Trade(models.Model):
    Trader = models.ForeignKey(User, on_delete=models.CASCADE)
    Stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    Trade_Type = models.CharField(max_length=4, choices=Trade_Choices)
    Trade_Qty = models.IntegerField()
    Trade_Price = models.FloatField(null=True)  # Adjusted field type
    Trade_Date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.Trade_Type} {self.Trade_Qty} {self.Stock} @{self.Trade_Price}"

class Stock_Portfolio(models.Model):
    Trader = models.ForeignKey(User, on_delete=models.CASCADE)
    Stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    Units = models.IntegerField()
    Invested = models.FloatField(null=True)  # Adjusted field type
    Current_Value = models.FloatField(null=True)  # Adjusted field type

    def __str__(self):
        return f"{self.Stock} gave gain/loss of {round(((self.Current_Value / self.Invested) - 1) * 100, 2)} %"

class Stock_Profit_Loss(models.Model):
    Trader = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    realized_profit_loss = models.FloatField(null=True,default=0)  # Adjusted field type

    def __str__(self):
        return f"{self.Trader.first_name} has made gains/loss of {self.realized_profit_loss} Rs."
