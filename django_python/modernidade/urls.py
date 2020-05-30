from django.urls import path


from .views import IndexView, database


urlpatterns = [
    #path('', views.post_list, name='post_list'),
    path('',IndexView.as_view(), name='index'),
    #path('database', database),
    #path('admin/',admin.site.urls),
    #path('', index),
    path('database',database)

    
]