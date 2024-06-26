# Generated by Django 4.1 on 2022-12-19 13:14

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("Broker", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Stock_Trade",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "Trade_Type",
                    models.CharField(
                        choices=[("Buy", "Buy"), ("Sell", "Sell")], max_length=4
                    ),
                ),
                (
                    "Trade_Method",
                    models.CharField(
                        choices=[("Market", "Market"), ("Limit", "Limit")],
                        max_length=10,
                    ),
                ),
                ("Trade_Qty", models.IntegerField()),
                ("Trade_Price", models.FloatField()),
                ("Trade_Date", models.DateTimeField(auto_now_add=True)),
                (
                    "Stock",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="Broker.stock"
                    ),
                ),
                (
                    "Trader",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Stock_Profit_Loss",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("realized_profit_loss", models.FloatField(default=0)),
                (
                    "Trader",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Stock_Portfolio",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("Units", models.IntegerField()),
                ("Invested", models.FloatField()),
                ("Current_Value", models.FloatField()),
                (
                    "Stock",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="Broker.stock"
                    ),
                ),
                (
                    "Trader",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
