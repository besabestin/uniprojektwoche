import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Life } from 'src/classes/story';
import { ChatResponse } from 'src/classes/chatresponse';
import { Observable, map, of } from 'rxjs';
import { Story } from 'src/classes/story';
import { SearchResult } from 'src/classes/story';

@Injectable({
  providedIn: 'root'
})
export class StoryService {
  url = "http://localhost:5152/stories";
  chatUrl = "http://localhost:5152/chat";
  newStoryUrl = "http://localhost:5152/newstory";
  searchUrl = "http://localhost:5152/searchjournal";

  aLife?: Life;

  chatMessages : string[] = [
    "Hello! I am your own personal assistant. Ask away anything"
  ];

  searchResult = ""

  constructor(private http: HttpClient) { }
  
  getStories(){
    this.http.get<Life>(this.url)
    .subscribe(theLife => {
      if(theLife){
        this.aLife = theLife;
      }
    })
  }

  addStory(storyDate: string, storyContent: string){
    //todo normally add the one coming from backend
    let story = {when: storyDate, content: storyContent}
    this.http.post<Story>(this.newStoryUrl, story, {'headers': { 'Content-Type': 'application/json'} })
    .subscribe(aStory => {
      if (aStory){
        console.log(aStory)
        this.aLife?.stories?.unshift(aStory);
      }
    })
  }

  getChatResponse(){
    this.http.post<ChatResponse>(this.chatUrl, {"messages": this.chatMessages}, {'headers': { 'Content-Type': 'application/json'} })
    .subscribe(aReply => {
      if(aReply){
        if (aReply.message !== '')
          this.chatMessages.push(aReply.message!)
      }
    })
  }

  searchInJournal(searchMessage: string){
    this.http.post<SearchResult>(this.searchUrl, {"entry": searchMessage}, {'headers': { 'Content-Type': 'application/json'} })
    .subscribe(aResult => {
      if(aResult && aResult.message !== ''){
        this.searchResult = aResult.message!;
      }
    })
  }
}
