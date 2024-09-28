from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

class Exchange(models.Model):
    Name = models.CharField(max_length=20)
    Extension = models.CharField(max_length=20)
    Currency = models.CharField(max_length=10)
    Country = models.CharField(max_length=10)
    Frac = models.IntegerField(null=True)

    def __str__(self):
        return self.Name


Cap_Choices = (("Large","Large"),("Mid","Mid"),("Small","Small"))

class Stock(models.Model):
    Exchange = models.ForeignKey(Exchange,on_delete=models.CASCADE)
    Sector = models.CharField(max_length=50)
    Cap = models.CharField(max_length=10, choices=Cap_Choices)  # Adjusted max_length
    Company = models.CharField(max_length=100)  # Adjusted max_length
    Symbol = models.CharField(max_length=16, primary_key=True)
    Display = models.CharField(max_length=16)
    CLS_Price = models.FloatField(null=True)
    EOD_Price = models.FloatField(null=True)  # Adjusted field type
    Expected_Price = ArrayField(models.FloatField(null=True))  # Adjusted field type
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
    trade_days = models.IntegerField(null=True)
    correct_reg = models.IntegerField(null=True)
    correct_class = models.IntegerField(null=True)
    reg_acc = models.FloatField(null=True)
    class_acc = models.FloatField(null=True)

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
    Units = models.FloatField()
    Invested = models.FloatField(null=True)  # Adjusted field type
    Current_Value = models.FloatField(null=True)  # Adjusted field type

    def __str__(self):
        return f"{self.Stock} gave gain/loss of {round(((self.Current_Value / self.Invested) - 1) * 100, 2)} %"

class Stock_Profit_Loss(models.Model):
    Trader = models.ForeignKey(User, on_delete=models.CASCADE)
    Exchange = models.ForeignKey(Exchange, on_delete=models.CASCADE)
    realized_profit_loss = models.FloatField(null=True, default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['Trader', 'Exchange'], name='unique_trader_exchange')
        ]

    def __str__(self):
        return f"{self.Trader.first_name} has made gains/loss of {self.realized_profit_loss} {self.Exchange.Currency} on {self.Exchange}"
