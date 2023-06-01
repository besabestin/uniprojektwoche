import { Component, OnInit } from '@angular/core';
import { Life } from 'src/classes/story';
import { StoryService } from '../story.service';

@Component({
  selector: 'app-diary',
  templateUrl: './diary.component.html',
  styleUrls: ['./diary.component.css']
})
export class DiaryComponent implements OnInit {

  storyDate = "";
  storyContent = "";
  searchEntry = "";

  constructor(public storyService : StoryService) { }

  ngOnInit(): void { }

  saveStory(){
    this.storyService.addStory(this.storyDate, this.storyContent);
    this.storyDate = "";
    this.storyContent = "";
  }

  searchInStory(){
    this.storyService.searchInJournal(this.searchEntry);
  }

}
